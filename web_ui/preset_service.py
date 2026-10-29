import re
import uuid
import copy
import streamlit as st
from web_ui.constants import GLOBAL_GROUP_ID
from web_ui.storage import write_presets_to_disk
from web_ui.state_sync import (
    apply_settings_from_template,
    update_session_state_from_preset,
    save_presets,
    sync_widgets_to_colors_list,
)


def extract_base_name(name):
    m = re.match(r"^(.*?)\s*\(Kopieren(?:\s+\d+)?\)$", name.strip())
    return m.group(1).strip() if m else name.strip()


def is_preset_template_master(preset_id):
    if not st.session_state.presets or preset_id == st.session_state.presets[0]["id"]:
        return False
    return any(p["values"].get("templateId") == preset_id for p in st.session_state.presets)


def get_preset_icon_tag(preset):
    if is_preset_template_master(preset["id"]):
        return " 👑"
    if preset["values"].get("templateId") is not None:
        return " 🔗"
    return ""


def get_group_name(group_id):
    if group_id == GLOBAL_GROUP_ID or group_id is None:
        return "Global"
    g = next((x for x in st.session_state.groups if x["id"] == group_id), None)
    return g["name"] if g else "Global"


def get_preset_name(preset_id):
    p = next((x for x in st.session_state.presets if x["id"] == preset_id), None)
    return p["name"] if p else str(preset_id)


def get_presets_in_group(group_id):
    if group_id == GLOBAL_GROUP_ID or group_id is None:
        return [p for p in st.session_state.presets if p.get("groupId") is None or p.get("groupId") == GLOBAL_GROUP_ID]
    return [p for p in st.session_state.presets if p.get("groupId") == group_id]


def generate_copy_name(original_name, existing_names):
    base_name = extract_base_name(original_name)

    existing_nums = set()
    for name in existing_names:
        if name == f"{base_name} (Kopieren)":
            existing_nums.add(1)
        else:
            m_ex = re.match(rf"^{re.escape(base_name)}\s*\(Kopieren\s+(\d+)\)$", name)
            if m_ex:
                existing_nums.add(int(m_ex.group(1)))

    if not existing_nums or 1 not in existing_nums:
        return f"{base_name} (Kopieren)"
    else:
        next_num = 2
        while next_num in existing_nums:
            next_num += 1
        return f"{base_name} (Kopieren {next_num})"


def select_preset_by_id(preset_id):
    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is not None:
        st.session_state.preset_index = idx
        st.session_state.preset_guid = preset_id
        gid = st.session_state.presets[idx].get("groupId")
        st.session_state.active_group_id = gid if gid else GLOBAL_GROUP_ID
        update_session_state_from_preset(st.session_state.presets[idx])
        write_presets_to_disk()


def copy_preset_by_id(preset_id, new_name=None):
    save_presets()

    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is None:
        return

    new_id = str(uuid.uuid4())
    copied_data = copy.deepcopy(st.session_state.presets[idx])
    copied_data["id"] = new_id

    existing_names = [p["name"] for p in st.session_state.presets]
    if new_name and new_name.strip():
        copied_data["name"] = new_name.strip()
    else:
        copied_data["name"] = generate_copy_name(copied_data["name"], existing_names)

    base_root = extract_base_name(st.session_state.presets[idx]["name"])

    insert_idx = idx + 1
    while insert_idx < len(st.session_state.presets):
        next_p = st.session_state.presets[insert_idx]
        if next_p.get("groupId") == copied_data.get("groupId"):
            m_next = re.match(rf"^{re.escape(base_root)}\s*\(Kopieren(?:\s+\d+)?\)$", next_p["name"])
            if m_next:
                insert_idx += 1
                continue
        break

    st.session_state.presets.insert(insert_idx, copied_data)
    select_preset_by_id(new_id)
    st.rerun()


def move_preset_in_group(preset_id, direction):
    preset = next((p for p in st.session_state.presets if p["id"] == preset_id), None)
    if not preset or preset["id"] == st.session_state.presets[0]["id"]:
        return

    group_id = preset.get("groupId") or GLOBAL_GROUP_ID
    siblings = get_presets_in_group(group_id)
    s_idx = next((i for i, p in enumerate(siblings) if p["id"] == preset_id), None)
    if s_idx is None:
        return

    target_s_idx = s_idx + direction
    if group_id == GLOBAL_GROUP_ID and target_s_idx == 0:
        return

    if 0 <= target_s_idx < len(siblings):
        target_preset = siblings[target_s_idx]
        p_idx = st.session_state.presets.index(preset)
        t_idx = st.session_state.presets.index(target_preset)
        st.session_state.presets[p_idx], st.session_state.presets[t_idx] = (
            st.session_state.presets[t_idx],
            st.session_state.presets[p_idx],
        )
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        write_presets_to_disk()


def unassign_preset(preset_id):
    preset = next((p for p in st.session_state.presets if p["id"] == preset_id), None)
    if not preset or preset["id"] == st.session_state.presets[0]["id"]:
        return

    cur_group = preset.get("groupId")
    if not cur_group or cur_group == GLOBAL_GROUP_ID:
        return

    cur_group_presets = get_presets_in_group(cur_group)
    prev_idx_in_group = next((i for i, p in enumerate(cur_group_presets) if p["id"] == preset_id), 0) - 1

    preset["groupId"] = None

    if st.session_state.preset_guid == preset_id:
        remaining = get_presets_in_group(cur_group)
        if len(remaining) > 0:
            target_idx = max(0, prev_idx_in_group)
            if target_idx >= len(remaining):
                target_idx = len(remaining) - 1
            select_preset_by_id(remaining[target_idx]["id"])
        else:
            select_preset_by_id(st.session_state.presets[0]["id"])
            st.session_state.active_group_id = cur_group
    else:
        write_presets_to_disk()


def delete_preset_by_id(preset_id):
    if len(st.session_state.presets) <= 1 or preset_id == st.session_state.presets[0]["id"]:
        return

    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is None:
        return

    cur_group_presets = get_presets_in_group(st.session_state.active_group_id)
    prev_idx_in_group = next((i for i, p in enumerate(cur_group_presets) if p["id"] == preset_id), 0) - 1

    removed_preset = st.session_state.presets.pop(idx)

    for pre in st.session_state.presets:
        preVals = pre["values"]
        if preVals.get("nextEffectId") == removed_preset["id"]:
            preVals["nextEffectId"] = None
            st.session_state.pop(f"next_effect_select_{pre['id']}", None)
        if preVals.get("templateId") == removed_preset["id"]:
            apply_settings_from_template(removed_preset["values"], preVals)
            st.session_state.pop(f"template_select_{pre['id']}", None)

    st.session_state.pop(f"next_effect_select_{removed_preset['id']}", None)
    st.session_state.pop(f"template_select_{removed_preset['id']}", None)
    if st.session_state.get("nextEffectId") == removed_preset["id"]:
        st.session_state.nextEffectId = None

    if st.session_state.preset_guid == preset_id:
        remaining_in_group = get_presets_in_group(st.session_state.active_group_id)
        if len(remaining_in_group) > 0:
            target_idx = max(0, prev_idx_in_group)
            if target_idx >= len(remaining_in_group):
                target_idx = len(remaining_in_group) - 1
            select_preset_by_id(remaining_in_group[target_idx]["id"])
        else:
            cur_group = st.session_state.active_group_id
            select_preset_by_id(st.session_state.presets[0]["id"])
            st.session_state.active_group_id = cur_group
    else:
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        write_presets_to_disk()


def move_group(group_id, direction):
    idx = next((i for i, g in enumerate(st.session_state.groups) if g["id"] == group_id), None)
    if idx is None:
        return
    new_idx = idx + direction
    if 0 <= new_idx < len(st.session_state.groups):
        st.session_state.groups[idx], st.session_state.groups[new_idx] = (
            st.session_state.groups[new_idx],
            st.session_state.groups[idx],
        )
        st.session_state.active_group_id = group_id
        st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1
        write_presets_to_disk()


def delete_group_by_id(group_id):
    to_delete = [p for p in st.session_state.presets if p.get("groupId") == group_id]
    to_delete_ids = {p["id"] for p in to_delete}
    if len(st.session_state.presets) > 0:
        to_delete_ids.discard(st.session_state.presets[0]["id"])

    for pre in st.session_state.presets:
        if pre["id"] not in to_delete_ids:
            preVals = pre["values"]
            if preVals.get("nextEffectId") in to_delete_ids:
                preVals["nextEffectId"] = None
                st.session_state.pop(f"next_effect_select_{pre['id']}", None)
            if preVals.get("templateId") in to_delete_ids:
                tmpl = next((p for p in to_delete if p["id"] == preVals.get("templateId")), None)
                if tmpl:
                    apply_settings_from_template(tmpl["values"], preVals)
                else:
                    preVals["templateId"] = None
                st.session_state.pop(f"template_select_{pre['id']}", None)

    for del_id in to_delete_ids:
        st.session_state.pop(f"next_effect_select_{del_id}", None)
        st.session_state.pop(f"template_select_{del_id}", None)

    if st.session_state.get("nextEffectId") in to_delete_ids:
        st.session_state.nextEffectId = None

    current_options = [GLOBAL_GROUP_ID] + [g["id"] for g in st.session_state.groups]
    prev_group_id = GLOBAL_GROUP_ID
    if group_id in current_options:
        curr_idx = current_options.index(group_id)
        prev_group_id = current_options[max(0, curr_idx - 1)]

    st.session_state.presets = [p for p in st.session_state.presets if p["id"] not in to_delete_ids]
    st.session_state.groups = [g for g in st.session_state.groups if g["id"] != group_id]
    st.session_state.active_group_id = prev_group_id
    st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1

    presets_in_target = get_presets_in_group(prev_group_id)
    if st.session_state.preset_guid in to_delete_ids or st.session_state.preset_index >= len(st.session_state.presets):
        if len(presets_in_target) > 0:
            select_preset_by_id(presets_in_target[0]["id"])
        else:
            st.session_state.preset_index = 0
            st.session_state.preset_guid = st.session_state.presets[0]["id"]
            update_session_state_from_preset(st.session_state.presets[0])
            write_presets_to_disk()
    else:
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        write_presets_to_disk()


def cb_add_color():
    sync_widgets_to_colors_list()
    st.session_state.colors.append({"color": "#ffffff", "threshold": 1.0})

    new_idx = len(st.session_state.colors) - 1
    st.session_state[f"color_{new_idx}"] = "#ffffff"
    st.session_state[f"threshold_{new_idx}"] = 1.0

    save_presets()


def remove_color(index):
    if len(st.session_state.colors) <= 2:
        return

    st.session_state.colors.pop(index)
    for i in range(index, len(st.session_state.colors) + 1):
        if f"color_{i + 1}" in st.session_state:
            st.session_state[f"color_{i}"] = st.session_state[f"color_{i + 1}"]

        if f"threshold_{i + 1}" in st.session_state:
            st.session_state[f"threshold_{i}"] = st.session_state[f"threshold_{i + 1}"]

    last_idx = len(st.session_state.colors)
    st.session_state.pop(f"color_{last_idx}", None)
    st.session_state.pop(f"threshold_{last_idx}", None)
    sync_widgets_to_colors_list()

    save_presets()


def move_color(index, direction):
    new_index = index + direction
    if index <= 0 or new_index <= 0 or index >= len(st.session_state.colors) or new_index >= len(st.session_state.colors):
        return

    sync_widgets_to_colors_list()
    st.session_state.colors[index]["color"], st.session_state.colors[new_index]["color"] = (
        st.session_state.colors[new_index]["color"],
        st.session_state.colors[index]["color"],
    )
    st.session_state[f"color_{index}"] = st.session_state.colors[index]["color"]
    st.session_state[f"color_{new_index}"] = st.session_state.colors[new_index]["color"]

    save_presets()
