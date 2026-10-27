import streamlit as st
from web_ui.constants import COLOR_MODES
from web_ui.preset_service import cb_add_color, move_color, remove_color
from web_ui.state_sync import ow_widget, save_presets, get_color_mode_id


def render_color_section():
    with st.expander("Farben", expanded=False):
        ow_widget(lambda: st.selectbox("Farbmodus", list(COLOR_MODES.values()), key="colorMode", on_change=save_presets), "colorMode")
        ow_widget(lambda: st.slider("Helligkeit", 0.0, 5.0, step=0.01, key="brightness", on_change=save_presets), "brightness")
        ow_widget(lambda: st.slider("Gamma", 0.0, 5.0, step=0.01, key="gamma", on_change=save_presets), "gamma")
        ow_widget(lambda: st.slider("Farbverstärkung", 0.1, 20.0, key="colorIncreaseFactor", on_change=save_presets), "colorIncreaseFactor")
        ow_widget(lambda: st.slider("Farbverlauf", 0.00, 0.50, key="colorTransition", on_change=save_presets), "colorTransition")

        if get_color_mode_id() not in [0, 6, 7, 8]:
            ow_widget(lambda: st.slider("Wert / Farb Verschmierung", 0.00, 1.00, step=0.01, key="valueColorBias", on_change=save_presets), "valueColorBias")
            ow_widget(lambda: st.toggle("Transparenz aus Wert", key="getAlphaFromValue", on_change=save_presets), "getAlphaFromValue", False)

        if get_color_mode_id() in [7, 8]:
            ow_widget(lambda: st.slider("Farbzyklus Geschwindigkeit", 0.00, 5.00, step=0.01, key="colorCycleSpeed", on_change=save_presets), "colorCycleSpeed")

        if get_color_mode_id() == 4:
            ow_widget(lambda: st.number_input("Farbwellen Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="colorWaveOrigin", on_change=save_presets), "colorWaveOrigin")
            ow_widget(lambda: st.number_input("Farbwellen Größe", min_value=1, max_value=999999, step=1, format="%d", key="colorWaveSize", on_change=save_presets), "colorWaveSize")
            ow_widget(lambda: st.slider("Farbwellen Geschwindigkeit", 1, 1000, key="colorWaveSpeed", on_change=save_presets), "colorWaveSpeed")
            ow_widget(lambda: st.toggle("Farbwellen Richtung (nach Außen / Innen)", key="colorWaveInwards", on_change=save_presets), "colorWaveInwards", False)

        ow_widget(lambda: st.slider("Zufallsfarben Faktor", 0.00, 1.00, step=0.01, key="noiseAmount", on_change=save_presets), "noiseAmount")
        ow_widget(lambda: st.slider("Zufallsfarben Glättung", 0.00, 1.00, step=0.01, key="noiseSmoothing", on_change=save_presets), "noiseSmoothing")
        ow_widget(lambda: st.toggle("Farbüberfluss", key="colorOverflow", on_change=save_presets), "colorOverflow", False)
        ow_widget(lambda: st.toggle("Regenbogen Farben", key="useRainbow", on_change=save_presets), "useRainbow", False)
        ow_widget(lambda: st.button("➕ Farbe hinzufügen", on_click=cb_add_color), "colors", False)

        for i, c in enumerate(st.session_state.colors):
            label = "Hintergrund Farbe" if i == 0 else f"Farbe {i}"

            if f"color_{i}" not in st.session_state:
                st.session_state[f"color_{i}"] = c["color"]
            if f"threshold_{i}" not in st.session_state:
                st.session_state[f"threshold_{i}"] = c["threshold"]

            col_color, col_actions = st.columns([1, 1], vertical_alignment="bottom", wrap=False)
            with col_color:
                st.color_picker(
                    label,
                    key=f"color_{i}",
                    on_change=save_presets,
                )
            with col_actions:
                has_actions = i > 0 and len(st.session_state.colors) > 2
                if has_actions:
                    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
                        if i > 1:
                            st.button("▲", key=f"mv_up_{i}", on_click=move_color, args=(i, -1))
                        if i > 0 and i < len(st.session_state.colors) - 1:
                            st.button("▼", key=f"mv_down_{i}", on_click=move_color, args=(i, 1))
                        if i > 0 and len(st.session_state.colors) > 2:
                            st.button("❌", key=f"rm_col_{i}", on_click=remove_color, args=(i,))

            if i >= len(st.session_state.colors) - 1:
                st.session_state[f"threshold_{i}"] = 1.0

            st.slider(
                "Grenzwert",
                min_value=0.00,
                max_value=1.00,
                step=0.01,
                key=f"threshold_{i}",
                disabled=(i >= len(st.session_state.colors) - 1),
                label_visibility="visible",
                on_change=save_presets,
            )
