import streamlit as st
from web_ui.constants import NONE_OPTION
from web_ui.preset_service import get_preset_name
from web_ui.state_sync import ow_widget, save_presets


def render_sequence_section():
    with st.expander("Sequenzierung", expanded=False):
        filtered_values = [v for v in st.session_state.presets if v["id"] != st.session_state.preset_guid]
        options = [NONE_OPTION] + [v["id"] for v in filtered_values]
        if "nextEffectId" not in st.session_state:
            st.session_state.nextEffectId = None

        cur_next_key = f"next_effect_select_{st.session_state.preset_guid}"
        cur_preset_for_next = st.session_state.presets[st.session_state.preset_index]
        linked_guid = cur_preset_for_next["values"].get("nextEffectId")

        default_index = 0
        if linked_guid is not None and linked_guid in options:
            default_index = options.index(linked_guid)

        def on_next_effect_changed():
            selected_guid = st.session_state.get(cur_next_key)
            if selected_guid == NONE_OPTION:
                st.session_state.nextEffectId = None
            else:
                st.session_state.nextEffectId = selected_guid

            save_presets()

        st.selectbox(
            "Nächster Effekt",
            options,
            index=default_index,
            format_func=lambda x: "Deaktiviert" if x is NONE_OPTION else get_preset_name(x),
            key=cur_next_key,
            on_change=on_next_effect_changed,
        )

        ow_widget(lambda: st.number_input("Effekt Dauer in ms", min_value=1, max_value=99999999, step=1, format="%d", key="effectDurationMs", on_change=save_presets), "effectDurationMs")
        ow_widget(lambda: st.number_input("Effekt Übergangsdauer in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionDurationMs", on_change=save_presets), "effectTransitionDurationMs")
        ow_widget(lambda: st.number_input("Übergangs Vorlaufzeit in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionWarmupDuration", on_change=save_presets), "effectTransitionWarmupDuration")
        ow_widget(lambda: st.toggle("Effekt nach Übergang zurücksetzen", key="resetEffectAfterTransition", on_change=save_presets), "resetEffectAfterTransition", False)
