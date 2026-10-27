import streamlit as st
from web_ui.state_sync import save_static_config


def render_global_section():
    with st.expander("Globale Einstellungen", expanded=False):
        st.number_input("FPS", min_value=1, max_value=400, step=1, format="%d", key="fps", on_change=save_static_config)
        st.number_input("LED Anzahl", min_value=2, max_value=99999, step=1, format="%d", key="ledCount", on_change=save_static_config)
