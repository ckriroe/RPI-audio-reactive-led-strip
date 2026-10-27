import streamlit as st
from web_ui.constants import AUDIO_MODES, MIN_VOLUME_DB, MAX_VOLUME_DB
from web_ui.state_sync import ow_widget, save_presets, get_audio_mode_id


def render_audio_section():
    with st.expander("Audio", expanded=False):
        ow_widget(lambda: st.selectbox("Audiomodus", list(AUDIO_MODES.values()), key="audioMode", on_change=save_presets), "audioMode")
        ow_widget(lambda: st.slider("Min. Lautstärke in dB", MIN_VOLUME_DB, MAX_VOLUME_DB, step=0.1, key="minFreqAmplitude", on_change=save_presets), "minFreqAmplitude")
        ow_widget(lambda: st.slider("Max. Lautstärke in dB", MIN_VOLUME_DB, MAX_VOLUME_DB, step=0.1, key="maxFreqAmplitude", on_change=save_presets), "maxFreqAmplitude")

        if get_audio_mode_id() == 0:
            ow_widget(lambda: st.slider("Vergleichswert Puffergröße", 1, 100, step=1, key="meanValueBufferSize", on_change=save_presets), "meanValueBufferSize")
            ow_widget(lambda: st.slider("Vergleichswert Grenzwert", 0.01, 1.0, step=0.01, key="meanValueThreshold", on_change=save_presets), "meanValueThreshold")

        if get_audio_mode_id() == 2:
            ow_widget(lambda: st.slider("Einfluss von Pegelveränderung", 0.0, 1.0, step=0.01, key="energyInfluence", on_change=save_presets), "energyInfluence")
            ow_widget(lambda: st.slider("Einfluss von Sprektrumsveränderung", 0.0, 1.0, step=0.01, key="fluxInfluence", on_change=save_presets), "fluxInfluence")

        ow_widget(lambda: st.slider("BPM Limit", -1, 999, step=1, key="bpmLimit", on_change=save_presets), "bpmLimit")
        ow_widget(lambda: st.slider("Audiowert Skalierung", 0.01, 5.00, step=0.01, key="audioResponseCurve", on_change=save_presets), "audioResponseCurve")
        ow_widget(lambda: st.slider("Min. Peakdauer in ms", 0, 999, step=1, key="audioPeakHoldTimeMs", on_change=save_presets), "audioPeakHoldTimeMs")
        ow_widget(lambda: st.number_input("Min. Frequenz in Hz", min_value=0, max_value=20000, step=1, format="%d", key="minFreq", on_change=save_presets), "minFreq")
        ow_widget(lambda: st.number_input("Max. Frequenz in Hz", min_value=0, max_value=20000, step=1, format="%d", key="maxFreq", on_change=save_presets), "maxFreq")
        ow_widget(lambda: st.number_input("Audio Puffer Größe", min_value=2, max_value=50000, step=1, format="%d", key="fftSize", on_change=save_presets), "fftSize")
