import streamlit as st
from web_ui.state_sync import ow_widget, save_presets


def render_pattern_section():
    with st.expander("Muster", expanded=False):
        ow_widget(lambda: st.slider("Effekt Wiederholungen", 1, 50, step=1, key="effectRepeats", on_change=save_presets), "effectRepeats")
        ow_widget(lambda: st.slider("Muster Teilungen", 0, 50, step=1, key="patternSplits", on_change=save_presets), "patternSplits")
        ow_widget(lambda: st.slider("Jedes N'te Muster verdrehen", -1, 50, step=1, key="patternFlip", on_change=save_presets), "patternFlip")
        ow_widget(lambda: st.number_input("Muster Verteilung", min_value=0, max_value=999999, step=1, format="%d", key="patternSpread", on_change=save_presets), "patternSpread")
        ow_widget(lambda: st.number_input("Muster Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="patternCenter", on_change=save_presets), "patternCenter")
        ow_widget(lambda: st.slider("Muster Größenveränderung", -5.00, 5.00, step=0.01, key="patternSectionSizeMod", on_change=save_presets), "patternSectionSizeMod")
