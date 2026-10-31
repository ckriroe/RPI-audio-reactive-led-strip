import copy
import json
import re
import uuid
import streamlit as st
from web_ui.constants import DYNAMIC_DEFAULTS, GLOBAL_GROUP_ID
from web_ui.migrations import migrate_frequency_amplitudes
from web_ui.storage import write_presets_to_disk
from web_ui.state_sync import save_presets
from web_ui.preset_service import (
    generate_copy_name,
    get_presets_in_group,
    select_preset_by_id,
)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>| ]', '_', name.strip())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned or "unnamed"


def export_preset(preset_id: str) -> tuple[str, str]:
    """Exports a single preset as a standalone JSON string.
    
    If the preset is a template slave, its inherited template values are flattened
    so that the preset is completely self-contained. nextEffectId and templateId are cleared.
    """
    if not st.session_state.presets or preset_id == st.session_state.presets[0]["id"]:
        raise ValueError("Standard-Voreinstellung kann nicht exportiert werden.")

    # If the preset to export is the currently active one, sync current UI state first
    if st.session_state.get("preset_guid") == preset_id:
        save_presets()

    preset = next((p for p in st.session_state.presets if p["id"] == preset_id), None)
    if not preset:
        raise ValueError(f"Voreinstellung '{preset_id}' nicht gefunden.")

    p_copy = copy.deepcopy(preset)
    tid = p_copy["values"].get("templateId")
    if tid:
        tmpl_preset = next((x for x in st.session_state.presets if x["id"] == tid), None)
        if tmpl_preset:
            merged = copy.deepcopy(DYNAMIC_DEFAULTS)
            merged.update(tmpl_preset["values"])
            merged.update(p_copy["values"])
            merged["templateId"] = None
            p_copy["values"] = merged
        else:
            p_copy["values"]["templateId"] = None

    p_copy["values"]["nextEffectId"] = None
    p_copy["values"]["templateId"] = None

    payload = {
        "type": "rpisc_preset",
        "version": 1,
        "preset": {
            "name": p_copy["name"],
            "values": p_copy["values"],
        },
    }
    filename = f"preset_{sanitize_filename(p_copy['name'])}.json"
    return json.dumps(payload, indent=2, ensure_ascii=False), filename


def export_group(group_id: str) -> tuple[str, str]:
    """Exports a whole group of presets as a JSON string.
    
    Internal relationships (templateId or nextEffectId pointing to presets within
    the same group) are preserved. External references pointing outside the group
    are flattened (for templates) or cleared (for nextEffectId).
    """
    if group_id == GLOBAL_GROUP_ID or group_id is None:
        raise ValueError("Globale Gruppe kann nicht als Gruppe exportiert werden.")

    group = next((g for g in st.session_state.groups if g["id"] == group_id), None)
    if not group:
        raise ValueError(f"Gruppe '{group_id}' nicht gefunden.")

    # If the active preset is in this group, sync current UI state first
    if st.session_state.get("active_group_id") == group_id and st.session_state.get("presets"):
        save_presets()

    group_presets = get_presets_in_group(group_id)
    group_preset_ids = {p["id"] for p in group_presets}

    exported_presets = []
    for p in group_presets:
        p_copy = copy.deepcopy(p)
        tid = p_copy["values"].get("templateId")
        if tid:
            if tid in group_preset_ids:
                # Internal template master is included in the export
                pass
            else:
                # External template master: flatten its values into this preset
                tmpl_preset = next((x for x in st.session_state.presets if x["id"] == tid), None)
                if tmpl_preset:
                    merged = copy.deepcopy(DYNAMIC_DEFAULTS)
                    merged.update(tmpl_preset["values"])
                    merged.update(p_copy["values"])
                    merged["templateId"] = None
                    p_copy["values"] = merged
                else:
                    p_copy["values"]["templateId"] = None

        nid = p_copy["values"].get("nextEffectId")
        if nid:
            if nid in group_preset_ids:
                # Internal next effect transition target is included
                pass
            else:
                # External target is not part of group export
                p_copy["values"]["nextEffectId"] = None

        # Clean up export object (keep internal id for cross-referencing within the group)
        exported_presets.append({
            "id": p_copy["id"],
            "name": p_copy["name"],
            "values": p_copy["values"],
        })

    payload = {
        "type": "rpisc_group",
        "version": 1,
        "group": {
            "name": group["name"],
            "presets": exported_presets,
        },
    }
    filename = f"group_{sanitize_filename(group['name'])}.json"
    return json.dumps(payload, indent=2, ensure_ascii=False), filename


def import_preset_to_group(
    data: dict,
    target_group_id: str | None,
    custom_name: str | None = None,
) -> tuple[bool, str, str | None]:
    """Imports a single preset into the target group.
    
    Generates a fresh UUID and disambiguates names if a collision occurs.
    """
    if not isinstance(data, dict):
        return False, "Ungültiges Dateiformat (kein JSON-Objekt).", None

    if data.get("type") == "rpisc_preset":
        preset_info = data.get("preset")
    elif "values" in data and "name" in data:
        preset_info = data
    else:
        return False, "Die ausgewählte Datei enthält keine gültige Voreinstellung.", None

    if not isinstance(preset_info, dict) or "values" not in preset_info:
        return False, "Voreinstellung enthält keine Einstellungsdaten.", None

    # Determine preset name
    name_candidate = custom_name.strip() if custom_name and custom_name.strip() else preset_info.get("name", "Importierte Voreinstellung")
    existing_names = [p["name"] for p in st.session_state.presets]
    final_name = generate_copy_name(name_candidate, existing_names) if name_candidate in existing_names else name_candidate

    # Ensure dynamic defaults are populated and external references cleared
    values = copy.deepcopy(DYNAMIC_DEFAULTS)
    values.update(preset_info.get("values", {}))
    values["templateId"] = None
    values["nextEffectId"] = None

    gid = target_group_id if (target_group_id and target_group_id != GLOBAL_GROUP_ID) else None
    new_pid = str(uuid.uuid4())

    new_preset = {
        "name": final_name,
        "id": new_pid,
        "groupId": gid,
        "values": values,
    }

    # Run through amplitude migrations to validate settings
    migrate_frequency_amplitudes({"presets": [new_preset]})

    st.session_state.presets.append(new_preset)
    if gid:
        st.session_state.active_group_id = gid
    else:
        st.session_state.active_group_id = GLOBAL_GROUP_ID

    select_preset_by_id(new_pid)
    write_presets_to_disk()

    return True, f"Voreinstellung '{final_name}' erfolgreich importiert!", new_pid


def import_group(
    data: dict,
    custom_name: str | None = None,
) -> tuple[bool, str, str | None]:
    """Imports a group of presets.
    
    Generates a new group UUID, fresh UUIDs for all presets, and remaps internal
    templateId and nextEffectId references so all internal relationships remain valid.
    """
    if not isinstance(data, dict):
        return False, "Ungültiges Dateiformat (kein JSON-Objekt).", None

    if data.get("type") == "rpisc_group":
        group_info = data.get("group")
    elif "presets" in data and "name" in data:
        group_info = data
    else:
        return False, "Die ausgewählte Datei enthält keine gültige Gruppe.", None

    if not isinstance(group_info, dict):
        return False, "Gruppe enthält keine gültigen Daten.", None

    # Determine group name
    grp_name_candidate = custom_name.strip() if custom_name and custom_name.strip() else group_info.get("name", "Importierte Gruppe")
    existing_grp_names = [g["name"] for g in st.session_state.groups] + ["Global"]
    final_grp_name = generate_copy_name(grp_name_candidate, existing_grp_names) if grp_name_candidate in existing_grp_names else grp_name_candidate

    new_gid = str(uuid.uuid4())
    raw_presets = group_info.get("presets", [])
    if not isinstance(raw_presets, list):
        raw_presets = []

    # Map old IDs to brand new UUIDs
    id_map = {
        p["id"]: str(uuid.uuid4())
        for p in raw_presets
        if isinstance(p, dict) and "id" in p
    }

    new_presets = []
    for p in raw_presets:
        if not isinstance(p, dict):
            continue
        old_id = p.get("id")
        new_pid = id_map.get(old_id, str(uuid.uuid4()))
        p_name = p.get("name", "Voreinstellung")
        existing_all = [x["name"] for x in st.session_state.presets] + [x["name"] for x in new_presets]
        final_p_name = generate_copy_name(p_name, existing_all) if p_name in existing_all else p_name

        vals = copy.deepcopy(p.get("values", {}))

        # Remap templateId if pointing to a preset within this group
        tid = vals.get("templateId")
        if tid and tid in id_map and tid != old_id:
            vals["templateId"] = id_map[tid]
        else:
            vals["templateId"] = None

        # Only standalone presets (templateId is None) need full dynamic defaults.
        # Template slaves must ONLY contain their overridden values.
        if vals["templateId"] is None:
            full_vals = copy.deepcopy(DYNAMIC_DEFAULTS)
            full_vals.update(vals)
            vals = full_vals

        # Remap nextEffectId if pointing to a preset within this group
        nid = vals.get("nextEffectId")
        if nid and nid in id_map:
            vals["nextEffectId"] = id_map[nid]
        else:
            vals["nextEffectId"] = None

        new_presets.append({
            "name": final_p_name,
            "id": new_pid,
            "groupId": new_gid,
            "values": vals,
        })

    # Run through amplitude migrations
    if new_presets:
        migrate_frequency_amplitudes({"presets": new_presets})

    st.session_state.groups.append({"id": new_gid, "name": final_grp_name})
    st.session_state.presets.extend(new_presets)
    st.session_state.active_group_id = new_gid
    st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1

    if new_presets:
        select_preset_by_id(new_presets[0]["id"])
    else:
        write_presets_to_disk()

    return True, f"Gruppe '{final_grp_name}' mit {len(new_presets)} Voreinstellung(en) erfolgreich importiert!", new_gid
