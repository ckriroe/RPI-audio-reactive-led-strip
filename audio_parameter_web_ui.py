import streamlit as st
from web_ui.constants import GLOBAL_GROUP_ID
from web_ui.storage import (
    load_version,
    load_presets,
    load_static_settings,
)
from web_ui.state_sync import (
    update_session_state_from_preset,
    update_session_state_from_static_settings,
)
from web_ui.styles import inject_custom_styles
from web_ui.dialogs import render_active_dialog
from web_ui.preset_service import get_presets_in_group
from web_ui.sections import (
    render_header,
    render_preset_manager,
    render_audio_section,
    render_effect_section,
    render_pattern_section,
    render_sequence_section,
    render_color_section,
    render_global_section,
)

app_version = load_version()
app_title = f"RPISC Einstellungen {app_version}" if app_version else "RPISC Einstellungen"

st.set_page_config(page_title=app_title, page_icon="🎛️", layout="centered")

preset_data = load_presets()
st.session_state.presets = preset_data["presets"]
st.session_state.groups = preset_data.get("groups", [])
st.session_state.preset_guid = preset_data["selectedPresetId"]

preset_index = 0
for i in range(len(st.session_state.presets)):
    if st.session_state.presets[i]["id"] == st.session_state.preset_guid:
        preset_index = i
        break

st.session_state.preset_index = preset_index
if st.session_state.preset_index >= len(st.session_state.presets):
    st.session_state.preset_index = 0
    st.session_state.preset_guid = st.session_state.presets[0]["id"]

if "active_group_id" not in st.session_state or st.session_state.active_group_id is None:
    cur_gid = st.session_state.presets[st.session_state.preset_index].get("groupId")
    st.session_state.active_group_id = cur_gid if cur_gid is not None else GLOBAL_GROUP_ID
elif st.session_state.active_group_id != GLOBAL_GROUP_ID:
    if not any(g["id"] == st.session_state.active_group_id for g in st.session_state.groups):
        st.session_state.active_group_id = GLOBAL_GROUP_ID

update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])

if "static_settings" not in st.session_state:
    st.session_state.static_settings = load_static_settings()
    update_session_state_from_static_settings(st.session_state.static_settings)

render_active_dialog()
inject_custom_styles()

render_header(app_title)
render_preset_manager()

presets_in_current_group = get_presets_in_group(st.session_state.active_group_id)
if len(presets_in_current_group) > 0:
    render_audio_section()
    render_effect_section()
    render_pattern_section()
    render_sequence_section()
    render_color_section()

render_global_section()
