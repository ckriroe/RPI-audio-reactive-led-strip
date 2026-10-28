import streamlit as st
from web_ui.preset_service import get_group_name, get_preset_icon_tag
from web_ui.dialogs import trigger_dialog


def render_header(app_title):
    col_title, col_shutdown = st.columns([8, 1], vertical_alignment="center", gap="small", wrap=False)
    with col_title:
        st.title(app_title)
    with col_shutdown:
        st.button(
            "⏻",
            key="btn_shutdown",
            help="System herunterfahren",
            on_click=trigger_dialog,
            args=("shutdown",),
        )

    if st.session_state.get("is_shutting_down"):
        st.error("**Das System wird heruntergefahren...**")

    active_p = st.session_state.presets[st.session_state.preset_index]
    active_gid = active_p.get("groupId")
    active_gname = get_group_name(active_gid)

    st.info(f"Aktive Voreinstellung: **{active_p['name']}{get_preset_icon_tag(active_p)}** (Gruppe: *{active_gname}*)")
