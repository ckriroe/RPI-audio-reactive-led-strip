import os
import json
import uuid
import copy
import streamlit as st
from web_ui.constants import (
    STATIC_CONFIG_FILE,
    PRESETS_FILE,
    VERSION_FILE,
    DYNAMIC_DEFAULTS,
    STATIC_DEFAULTS,
)
from web_ui.migrations import migrate_presets


def load_version():
    candidates = [
        VERSION_FILE,
        os.path.join("AuidoReactiveLedStrip", "Application", "version.txt"),
        os.path.join(os.path.dirname(__file__), "..", "AuidoReactiveLedStrip", "Application", "version.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ver = f.read().strip()
                    if ver:
                        return ver
            except Exception:
                pass
    return None


def load_presets():
    if os.path.exists(PRESETS_FILE):
        print("Loading preset file...\n")
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            jsn = json.load(f)
            jsn = migrate_presets(jsn)
            print("Preset file loaded:\n", jsn, "\n")
            return jsn
    else:
        print("Preset file did not yet exist, creating presets file\n")
        return create_default_presets_file()


def load_static_settings():
    if os.path.exists(STATIC_CONFIG_FILE):
        print("Loading static config file...\n")
        with open(STATIC_CONFIG_FILE, "r", encoding="utf-8") as f:
             jsn = json.load(f)
             print("Static config file loaded:\n", jsn, "\n")
             return jsn
    else:
        print("Static config file did not yet exist, creating static config file\n")
        return create_default_static_settings_file()


def create_default_presets_file():
    preset_id = str(uuid.uuid4())
    initial_data = {
        "selectedPresetId": preset_id,
        "groups": [],
        "presets": [
            {
                "name": "Standard",
                "id": preset_id,
                "groupId": None,
                "values": copy.deepcopy(DYNAMIC_DEFAULTS),
            }
        ],
    }
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(initial_data, f, indent=2)

    return initial_data


def create_default_static_settings_file():
    initial_data = copy.deepcopy(STATIC_DEFAULTS)
    write_static_config_to_disk(initial_data)
    return initial_data


def write_presets_to_disk():
    data = {
        "selectedPresetId": st.session_state.preset_guid,
        "groups": st.session_state.get("groups", []),
        "presets": st.session_state.presets,
    }
    print("Writing presets to disk:\n", data, "\n")
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_static_config_to_disk(values):
    print("Writing static config to disk:\n", values, "\n")
    with open(STATIC_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)
