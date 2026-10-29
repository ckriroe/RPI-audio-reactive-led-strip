import streamlit as st
from web_ui.dialogs import trigger_dialog


def render_header(app_title):
    if "toast_message" in st.session_state:
        t_msg, t_icon = st.session_state.pop("toast_message")
        st.toast(t_msg, icon=t_icon)

    pop_ver = st.session_state.get("popover_ver", 0)
    col_title, col_shutdown, col_menu = st.columns([8, 1, 1], vertical_alignment="center", gap="small", wrap=False)
    with col_title:
        st.title(app_title)
    with col_shutdown:
        st.button(
            "🔌",
            key="btn_shutdown",
            on_click=trigger_dialog,
            args=("shutdown",),
        )
    with col_menu:
        with st.popover("", key=f"pop_header_menu_{pop_ver}"):
            st.button(
                "💾 Backup erstellen",
                key="btn_create_backup",
                on_click=trigger_dialog,
                args=("create_backup",),
                use_container_width=True,
            )
            st.button(
                "📥 Backup wiederherstellen",
                key="btn_import_backup",
                on_click=trigger_dialog,
                args=("import_backup",),
                use_container_width=True,
            )
            st.button(
                "🔄 Nach Updates suchen",
                key="btn_check_update",
                on_click=trigger_dialog,
                args=("check_update",),
                use_container_width=True,
            )

    if st.session_state.get("is_shutting_down"):
        st.error("**Das System wird heruntergefahren...**")

    active_p = st.session_state.presets[st.session_state.preset_index]
    st.info(f"Aktive Voreinstellung: **{active_p['name']}**")
