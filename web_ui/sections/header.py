import streamlit as st
from web_ui.preset_service import get_group_name, get_preset_icon_tag


def render_header(app_title):
    st.title(app_title)

    active_p = st.session_state.presets[st.session_state.preset_index]
    active_gid = active_p.get("groupId")
    active_gname = get_group_name(active_gid)

    st.info(f"Aktive Voreinstellung: **{active_p['name']}{get_preset_icon_tag(active_p)}** (Gruppe: *{active_gname}*)")
