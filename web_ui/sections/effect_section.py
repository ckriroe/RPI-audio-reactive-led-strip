import streamlit as st
from web_ui.constants import EFFECT_MODES
from web_ui.state_sync import ow_widget, save_presets, get_effect_mode_id


def render_effect_section():
    with st.expander("Effekt", expanded=False):
        ow_widget(lambda: st.selectbox("Effektmodus", list(EFFECT_MODES.values()), key="effectMode", on_change=save_presets), "effectMode")
        ow_widget(lambda: st.number_input("Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="effectOrigin", on_change=save_presets), "effectOrigin")
        ow_widget(lambda: st.slider("Effekt Beschleunigung", 0.0, 5.0, step=0.01, key="acceleration", on_change=save_presets), "acceleration")

        if get_effect_mode_id() in (2, 4, 9, 10, 11, 12, 13, 14):
            ow_widget(lambda: st.slider("Effekt Geschwindigkeit", 1, 1000, key="speed", on_change=save_presets), "speed")

        ow_widget(lambda: st.slider("Effekt Verstärkung", 0.1, 10.0, key="valueIncreaseFactor", on_change=save_presets), "valueIncreaseFactor")

        if get_effect_mode_id() == 3:
            ow_widget(lambda: st.toggle("Statisches Spektrum", key="staticSpectrum", on_change=save_presets), "staticSpectrum", False)
            ow_widget(lambda: st.slider("Spektrum Sektionen", 1, 50, step=1, key="spectrumSections", on_change=save_presets), "spectrumSections")

        if get_effect_mode_id() in (4, 9, 12, 13, 14):
            ow_widget(lambda: st.slider("Partikel Größe", 1, 1000, step=1, key="particleSize", on_change=save_presets), "particleSize")

        if get_effect_mode_id() in (2, 9, 13):
            ow_widget(lambda: st.toggle("Abprallen", key="bouncyWave", on_change=save_presets), "bouncyWave", False)

        if get_effect_mode_id() == 9:
            ow_widget(lambda: st.slider("Max. Wellenanzahl", 1, 10, step=1, key="maxWaves", on_change=save_presets), "maxWaves")

        if get_effect_mode_id() in (2, 8, 14):
            ow_widget(lambda: st.slider("Verblassung nach Zeit", -1.0, 1.0, step=0.001, key="fadeOverTime", on_change=save_presets), "fadeOverTime")

        ow_widget(lambda: st.slider("Verblassung", 0.001, 0.999, step=0.001, key="fade", on_change=save_presets), "fade")
        ow_widget(lambda: st.slider("Sättigung", 0.01, 1.0, step=0.01, key="saturate", on_change=save_presets), "saturate")
        ow_widget(lambda: st.slider("Sättigungs Grenzwert", 0.0, 1.0, step=0.01, key="saturateThreshold", on_change=save_presets), "saturateThreshold")
