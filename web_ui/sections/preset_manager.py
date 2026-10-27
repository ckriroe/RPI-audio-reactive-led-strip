import uuid
import copy
import streamlit as st
from web_ui.constants import GLOBAL_GROUP_ID, NONE_OPTION, DYNAMIC_DEFAULTS
from web_ui.storage import write_presets_to_disk
from web_ui.state_sync import save_presets
from web_ui.dialogs import trigger_dialog
from web_ui.preset_service import (
    get_presets_in_group,
    get_preset_name,
    get_preset_icon_tag,
    is_preset_template_master,
    generate_copy_name,
    select_preset_by_id,
    move_preset_in_group,
    move_group,
    unassign_preset,
)


def render_preset_manager():
    group_options = [GLOBAL_GROUP_ID] + [g["id"] for g in st.session_state.groups]
    group_names = {GLOBAL_GROUP_ID: "Global", **{g["id"]: g["name"] for g in st.session_state.groups}}

    name_to_gids = {}
    for gid in group_options:
        name = group_names.get(gid, "Global")
        name_to_gids.setdefault(name, []).append(gid)

    group_display_map = {}
    for name, gids in name_to_gids.items():
        if len(gids) == 1:
            group_display_map[gids[0]] = name
        else:
            sorted_gids = sorted(gids)
            for count, gid in enumerate(sorted_gids):
                group_display_map[gid] = name + ("\u200b" * count)

    grp_ver = st.session_state.get("group_selector_ver", 0)
    grp_sel_key = f"group_selector_{grp_ver}"

    def on_group_change():
        selected_gid = st.session_state.get(grp_sel_key)
        if not selected_gid:
            selected_gid = GLOBAL_GROUP_ID
            st.session_state[grp_sel_key] = GLOBAL_GROUP_ID
        st.session_state.active_group_id = selected_gid
        presets_in_new_group = get_presets_in_group(selected_gid)
        cur_preset = st.session_state.presets[st.session_state.preset_index]
        cur_p_gid = cur_preset.get("groupId") or GLOBAL_GROUP_ID
        if cur_p_gid != selected_gid:
            if len(presets_in_new_group) > 0:
                select_preset_by_id(presets_in_new_group[0]["id"])

    grp_idx = 0
    if st.session_state.active_group_id in group_options:
        grp_idx = group_options.index(st.session_state.active_group_id)
    else:
        st.session_state.active_group_id = GLOBAL_GROUP_ID

    if st.session_state.get(grp_sel_key) not in group_options or st.session_state.get(grp_sel_key) != st.session_state.active_group_id:
        st.session_state[grp_sel_key] = st.session_state.active_group_id
        grp_idx = group_options.index(st.session_state.active_group_id)

    with st.expander("📁 Voreinstellungen & Gruppen", expanded=True):
        col_grp, col_grp_menu = st.columns([8, 1], vertical_alignment="bottom", gap="small", wrap=False)
        with col_grp:
            st.selectbox(
                "Gruppe",
                group_options,
                index=grp_idx,
                format_func=lambda x: group_display_map.get(x, "Global"),
                key=grp_sel_key,
                on_change=on_group_change,
            )
        pop_ver = st.session_state.get("popover_ver", 0)
        with col_grp_menu:
            is_custom_group = (st.session_state.active_group_id != GLOBAL_GROUP_ID and st.session_state.active_group_id is not None)
            with st.popover("", key=f"pop_grp_menu_{pop_ver}"):
                st.button("➕ Neue Gruppe", key="btn_grp_new", on_click=trigger_dialog, args=("create_group",), use_container_width=True)

                cur_grp_idx = 0
                if is_custom_group:
                    cur_grp_idx = next((i for i, g in enumerate(st.session_state.groups) if g["id"] == st.session_state.active_group_id), 0)

                st.button(
                    "▲ Nach oben",
                    key="btn_grp_up",
                    disabled=(not is_custom_group or cur_grp_idx == 0),
                    on_click=move_group,
                    args=(st.session_state.active_group_id, -1),
                    use_container_width=True,
                )
                st.button(
                    "▼ Nach unten",
                    key="btn_grp_down",
                    disabled=(not is_custom_group or cur_grp_idx == len(st.session_state.groups) - 1),
                    on_click=move_group,
                    args=(st.session_state.active_group_id, 1),
                    use_container_width=True,
                )
                st.button(
                    "✏️ Gruppe umbenennen",
                    key="btn_grp_rename",
                    disabled=not is_custom_group,
                    on_click=trigger_dialog,
                    args=("rename_group", st.session_state.active_group_id),
                    use_container_width=True,
                )
                st.button(
                    "🗑️ Gruppe löschen",
                    key="btn_grp_delete",
                    disabled=not is_custom_group,
                    on_click=trigger_dialog,
                    args=("delete_group", st.session_state.active_group_id),
                    use_container_width=True,
                )

        if is_custom_group:
            ungrouped_presets = [p for p in get_presets_in_group(GLOBAL_GROUP_ID) if p["id"] != st.session_state.presets[0]["id"]]
            if len(ungrouped_presets) > 0:
                def on_add_preset_from_global():
                    selected_pid = st.session_state.add_from_global_selector
                    if selected_pid and selected_pid != NONE_OPTION:
                        p = next((x for x in st.session_state.presets if x["id"] == selected_pid), None)
                        if p:
                            p["groupId"] = st.session_state.active_group_id
                            select_preset_by_id(selected_pid)
                        st.session_state.add_from_global_selector = NONE_OPTION

                options = [NONE_OPTION] + [p["id"] for p in ungrouped_presets]
                format_map = {NONE_OPTION: "Voreinstellung hinzufügen", **{p["id"]: p["name"] for p in ungrouped_presets}}
                st.selectbox(
                    "➕ Globale Voreinstellung hinzufügen",
                    options,
                    index=0,
                    format_func=lambda x: format_map.get(x, str(x)),
                    key="add_from_global_selector",
                    on_change=on_add_preset_from_global,
                )

        presets_in_current_group = get_presets_in_group(st.session_state.active_group_id)
        current_grp_title = group_names.get(st.session_state.active_group_id, "Global")

        with st.expander(f"📂 Voreinstellungen in {current_grp_title} ({len(presets_in_current_group)})", expanded=True):
            if len(presets_in_current_group) == 0:
                st.info("Diese Gruppe enthält noch keine Voreinstellungen.")
                if is_custom_group:
                    if st.button("➕ Neue Voreinstellung erstellen", key="btn_create_empty_grp_pre"):
                        new_id = str(uuid.uuid4())
                        existing_names = [p["name"] for p in st.session_state.presets]
                        new_name = generate_copy_name("Standard", existing_names)
                        new_preset = {
                            "name": new_name,
                            "id": new_id,
                            "groupId": st.session_state.active_group_id,
                            "values": copy.deepcopy(DYNAMIC_DEFAULTS),
                        }
                        st.session_state.presets.append(new_preset)
                        select_preset_by_id(new_id)
                        st.rerun()
            else:
                for idx_in_grp, p in enumerate(presets_in_current_group):
                    is_standard = (p["id"] == st.session_state.presets[0]["id"])
                    is_active = (p["id"] == st.session_state.preset_guid)
                    icon_tag = get_preset_icon_tag(p)

                    col_name, col_actions = st.columns([7, 3], vertical_alignment="center", gap="small", wrap=False)
                    with col_name:
                        if is_active:
                            st.markdown(f"**🟢 {p['name']}{icon_tag}**")
                        else:
                            st.markdown(f"⚪ {p['name']}{icon_tag}")
                    with col_actions:
                        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
                            if not is_active:
                                st.button(
                                    "✅",
                                    key=f"btn_pre_sel_{p['id']}",
                                    on_click=select_preset_by_id,
                                    args=(p["id"],),
                                    help="Voreinstellung aktivieren",
                                )
                            with st.popover("", key=f"pop_pre_{p['id']}_{pop_ver}"):
                                can_move_up = False
                                if not is_standard:
                                    if st.session_state.active_group_id == GLOBAL_GROUP_ID:
                                        can_move_up = (idx_in_grp > 1)
                                    else:
                                        can_move_up = (idx_in_grp > 0)

                                can_move_down = (not is_standard and idx_in_grp < len(presets_in_current_group) - 1)

                                st.button(
                                    "▲ Nach oben",
                                    key=f"btn_up_{p['id']}",
                                    disabled=(not can_move_up),
                                    on_click=move_preset_in_group,
                                    args=(p["id"], -1),
                                    use_container_width=True,
                                )
                                st.button(
                                    "▼ Nach unten",
                                    key=f"btn_down_{p['id']}",
                                    disabled=(not can_move_down),
                                    on_click=move_preset_in_group,
                                    args=(p["id"], 1),
                                    use_container_width=True,
                                )
                                st.button(
                                    "✏️ Umbenennen",
                                    key=f"btn_ren_{p['id']}",
                                    disabled=is_standard,
                                    on_click=trigger_dialog,
                                    args=("rename_preset", p["id"]),
                                    use_container_width=True,
                                )
                                st.button(
                                    "📋 Kopieren",
                                    key=f"btn_cp_{p['id']}",
                                    on_click=trigger_dialog,
                                    args=("copy_preset", p["id"]),
                                    use_container_width=True,
                                )
                                if p.get("groupId") is not None and p.get("groupId") != GLOBAL_GROUP_ID:
                                    st.button(
                                        "↩️ In 'Global' verschieben",
                                        key=f"btn_ungrp_{p['id']}",
                                        on_click=unassign_preset,
                                        args=(p["id"],),
                                        use_container_width=True,
                                    )
                                st.button(
                                    "🗑️ Löschen",
                                    key=f"btn_del_{p['id']}",
                                    disabled=is_standard,
                                    on_click=trigger_dialog,
                                    args=("delete_preset", p["id"]),
                                    use_container_width=True,
                                )

        current_guid = st.session_state.preset_guid
        is_master = is_preset_template_master(current_guid)
        cur_p = st.session_state.presets[st.session_state.preset_index]
        cur_p_gid = cur_p.get("groupId") or GLOBAL_GROUP_ID
        if st.session_state.preset_index != 0 and not is_master and len(presets_in_current_group) > 0 and cur_p_gid == st.session_state.active_group_id:
            std_id = st.session_state.presets[0]["id"]
            filtered_values = [
                v for v in st.session_state.presets
                if v["id"] != st.session_state.preset_guid
                and v["id"] != std_id
                and v["values"].get("templateId") is None
            ]
            options = [NONE_OPTION] + [v["id"] for v in filtered_values]
            if "templateId" not in st.session_state:
                st.session_state.templateId = None

            cur_tmpl_key = f"template_select_{st.session_state.preset_guid}"
            template_guid = cur_p["values"].get("templateId")

            default_index = 0
            if template_guid is not None and template_guid in options:
                default_index = options.index(template_guid)

            def on_template_changed():
                selected_guid = st.session_state.get(cur_tmpl_key)
                if selected_guid == NONE_OPTION:
                    st.session_state.templateId = None
                else:
                    st.session_state.templateId = selected_guid

                save_presets()

            st.selectbox(
                "Vorlage für aktive Voreinstellung",
                options,
                index=default_index,
                format_func=lambda x: "Deaktiviert" if x is NONE_OPTION else get_preset_name(x),
                key=cur_tmpl_key,
                on_change=on_template_changed,
            )
