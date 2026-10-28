import copy
import streamlit as st
from web_ui.constants import (
    AUDIO_MODES,
    EFFECT_MODES,
    COLOR_MODES,
    DYNAMIC_DEFAULTS,
    STATIC_DEFAULTS,
)
from web_ui.storage import (
    write_presets_to_disk,
    write_static_config_to_disk,
    load_static_settings,
)


def get_audio_mode_id():
    return next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode)


def get_effect_mode_id():
    return next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)


def get_color_mode_id():
    return next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)


def sync_widgets_to_colors_list():
    if "colors" in st.session_state:
        for i in range(len(st.session_state.colors)):
            c_key = f"color_{i}"
            t_key = f"threshold_{i}"

            if c_key in st.session_state:
                st.session_state.colors[i]["color"] = st.session_state[c_key]

            if t_key in st.session_state:
                st.session_state.colors[i]["threshold"] = st.session_state[t_key]


def apply_settings_from_template(template_values, preset_values):
    prev_preset_values = preset_values.copy()
    preset_values.clear()
    preset_values.update(template_values)
    preset_values.update(prev_preset_values)
    preset_values["templateId"] = None


def set_value_from_state2(stateObj, get_for_template, template_values, preset_key, session_key):
    get_for_template = get_for_template and template_values is not None
    if st.session_state.get("templateId") is not None:
        if st.session_state.get(preset_key + "_ow") is True:
            if get_for_template is True:
                stateObj[preset_key] = template_values.get(preset_key, DYNAMIC_DEFAULTS.get(preset_key))
            else:
                stateObj[preset_key] = st.session_state[session_key]
        elif get_for_template is True:
            stateObj[preset_key] = st.session_state[session_key]
    else:
        stateObj[preset_key] = st.session_state[session_key]


def set_value_from_state(stateObj, get_for_template, template_values, key):
    set_value_from_state2(stateObj, get_for_template, template_values, key, key)


def set_none_templateable_value_from_state(stateObj, get_for_template, template_values, key):
    if get_for_template is True:
        if template_values is not None:
            stateObj[key] = template_values.get(key, DYNAMIC_DEFAULTS.get(key))
    else:
        stateObj[key] = st.session_state.get(key, DYNAMIC_DEFAULTS.get(key))


def get_current_preset_values_from_state(get_for_template):
    sync_widgets_to_colors_list()

    st.session_state.audioModeValue = next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode)
    st.session_state.effectModeValue = next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
    st.session_state.colorModeValue = next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)

    template_values = None
    templateId = st.session_state.get("templateId")
    if templateId is not None:
        template_values = next((pre["values"] for pre in st.session_state.presets if pre["id"] == templateId), None)

    stateObj = {
        "templateId": st.session_state.templateId
    }

    if get_for_template is True:
        stateObj["templateId"] = None

    set_value_from_state(stateObj, get_for_template, template_values, "colors")
    set_value_from_state(stateObj, get_for_template, template_values, "useRainbow")
    set_value_from_state(stateObj, get_for_template, template_values, "effectOrigin")
    set_value_from_state(stateObj, get_for_template, template_values, "speed")
    set_value_from_state(stateObj, get_for_template, template_values, "minFreq")
    set_value_from_state(stateObj, get_for_template, template_values, "maxFreq")
    set_value_from_state(stateObj, get_for_template, template_values, "fade")
    set_value_from_state(stateObj, get_for_template, template_values, "fadeOverTime")
    set_value_from_state(stateObj, get_for_template, template_values, "bouncyWave")
    set_value_from_state(stateObj, get_for_template, template_values, "staticSpectrum")
    set_value_from_state(stateObj, get_for_template, template_values, "spectrumSections")
    set_value_from_state(stateObj, get_for_template, template_values, "saturate")
    set_value_from_state(stateObj, get_for_template, template_values, "saturateThreshold")
    set_value_from_state(stateObj, get_for_template, template_values, "meanValueBufferSize")
    set_value_from_state(stateObj, get_for_template, template_values, "meanValueThreshold")
    set_value_from_state2(stateObj, get_for_template, template_values, "audioMode", "audioModeValue")
    set_value_from_state2(stateObj, get_for_template, template_values, "effectMode", "effectModeValue")
    set_value_from_state2(stateObj, get_for_template, template_values, "colorMode", "colorModeValue")
    set_value_from_state(stateObj, get_for_template, template_values, "minFreqAmplitude")
    set_value_from_state(stateObj, get_for_template, template_values, "maxFreqAmplitude")
    set_value_from_state(stateObj, get_for_template, template_values, "colorIncreaseFactor")
    set_value_from_state(stateObj, get_for_template, template_values, "valueIncreaseFactor")
    set_value_from_state(stateObj, get_for_template, template_values, "colorTransition")
    set_value_from_state(stateObj, get_for_template, template_values, "colorCycleSpeed")
    set_value_from_state(stateObj, get_for_template, template_values, "valueColorBias")
    set_value_from_state(stateObj, get_for_template, template_values, "colorWaveOrigin")
    set_value_from_state(stateObj, get_for_template, template_values, "colorWaveSpeed")
    set_value_from_state(stateObj, get_for_template, template_values, "colorWaveSize")
    set_value_from_state(stateObj, get_for_template, template_values, "colorWaveInwards")
    set_value_from_state(stateObj, get_for_template, template_values, "noiseAmount")
    set_value_from_state(stateObj, get_for_template, template_values, "noiseSmoothing")
    set_value_from_state(stateObj, get_for_template, template_values, "getAlphaFromValue")
    set_value_from_state(stateObj, get_for_template, template_values, "colorOverflow")
    set_value_from_state(stateObj, get_for_template, template_values, "brightness")
    set_value_from_state(stateObj, get_for_template, template_values, "gamma")
    set_value_from_state(stateObj, get_for_template, template_values, "effectRepeats")
    set_value_from_state(stateObj, get_for_template, template_values, "acceleration")
    set_value_from_state(stateObj, get_for_template, template_values, "particleSize")
    set_value_from_state(stateObj, get_for_template, template_values, "patternSplits")
    set_value_from_state(stateObj, get_for_template, template_values, "patternSpread")
    set_value_from_state(stateObj, get_for_template, template_values, "patternFlip")
    set_value_from_state(stateObj, get_for_template, template_values, "patternCenter")
    set_value_from_state(stateObj, get_for_template, template_values, "patternSectionSizeMod")
    set_value_from_state(stateObj, get_for_template, template_values, "fftSize")
    set_value_from_state(stateObj, get_for_template, template_values, "bpmLimit")
    set_value_from_state(stateObj, get_for_template, template_values, "audioResponseCurve")
    set_value_from_state(stateObj, get_for_template, template_values, "audioPeakHoldTimeMs")
    set_value_from_state(stateObj, get_for_template, template_values, "maxWaves")
    set_none_templateable_value_from_state(stateObj, get_for_template, template_values, "nextEffectId")
    set_value_from_state(stateObj, get_for_template, template_values, "effectDurationMs")
    set_value_from_state(stateObj, get_for_template, template_values, "effectTransitionDurationMs")
    set_value_from_state(stateObj, get_for_template, template_values, "effectTransitionWarmupDuration")
    set_value_from_state(stateObj, get_for_template, template_values, "resetEffectAfterTransition")
    set_value_from_state(stateObj, get_for_template, template_values, "fluxInfluence")
    set_value_from_state(stateObj, get_for_template, template_values, "energyInfluence")

    return stateObj


def update_session_value_from_preset2(preset_values, template_values, preset_key, session_key):
    preset_value = preset_values.get(preset_key)
    if template_values is not None:
        if preset_key in preset_values:
            st.session_state[preset_key + "_ow"] = True
        else:
            preset_value = template_values.get(preset_key)
            st.session_state[preset_key + "_ow"] = False

    if preset_value is None:
        preset_value = DYNAMIC_DEFAULTS[preset_key]

    st.session_state[session_key] = preset_value


def update_session_value_from_preset(preset_values, template_values, key):
    update_session_value_from_preset2(preset_values, template_values, key, key)


def update_session_none_templateable_value_from_preset(preset_values, key):
    preset_value = preset_values.get(key)
    if key == "nextEffectId" and preset_value is not None:
        all_ids = {p["id"] for p in st.session_state.presets} if "presets" in st.session_state else set()
        if preset_value == st.session_state.get("preset_guid") or preset_value not in all_ids:
            preset_value = None
            preset_values["nextEffectId"] = None
    if preset_value is None:
        preset_value = DYNAMIC_DEFAULTS[key]

    st.session_state[key] = preset_value


def update_session_state_from_preset(preset_data):
    st.session_state.preset_guid = preset_data["id"]
    preset_values = preset_data["values"]

    template_values = None
    templateId = preset_values.get("templateId")
    if templateId is not None:
        template_preset = next((pre for pre in st.session_state.presets if pre["id"] == templateId), None)
        if template_preset is not None and templateId != preset_data["id"] and template_preset["values"].get("templateId") is None:
            template_values = template_preset["values"]
        else:
            preset_values["templateId"] = None
            templateId = None

    st.session_state.templateId = templateId

    if template_values is None:
        for k in DYNAMIC_DEFAULTS.keys():
            if k != "templateId":
                st.session_state[k + "_ow"] = False

    update_session_value_from_preset2(preset_values, template_values, "colors", "colorsRaw")

    st.session_state.colors = copy.deepcopy(st.session_state.colorsRaw)
    for i, c in enumerate(st.session_state.colors):
        st.session_state[f"color_{i}"] = c["color"]
        st.session_state[f"threshold_{i}"] = c["threshold"]

    update_session_value_from_preset(preset_values, template_values, "useRainbow")
    update_session_value_from_preset(preset_values, template_values, "effectOrigin")
    update_session_value_from_preset(preset_values, template_values, "speed")
    update_session_value_from_preset(preset_values, template_values, "minFreq")
    update_session_value_from_preset(preset_values, template_values, "maxFreq")
    update_session_value_from_preset(preset_values, template_values, "fade")
    update_session_value_from_preset(preset_values, template_values, "fadeOverTime")
    update_session_value_from_preset(preset_values, template_values, "bouncyWave")
    update_session_value_from_preset(preset_values, template_values, "staticSpectrum")
    update_session_value_from_preset(preset_values, template_values, "spectrumSections")
    update_session_value_from_preset(preset_values, template_values, "saturate")
    update_session_value_from_preset(preset_values, template_values, "saturateThreshold")
    update_session_value_from_preset(preset_values, template_values, "meanValueBufferSize")
    update_session_value_from_preset(preset_values, template_values, "meanValueThreshold")
    update_session_value_from_preset(preset_values, template_values, "minFreqAmplitude")
    update_session_value_from_preset(preset_values, template_values, "maxFreqAmplitude")
    update_session_value_from_preset(preset_values, template_values, "colorIncreaseFactor")
    update_session_value_from_preset(preset_values, template_values, "valueIncreaseFactor")
    update_session_value_from_preset(preset_values, template_values, "colorTransition")
    update_session_value_from_preset(preset_values, template_values, "colorCycleSpeed")
    update_session_value_from_preset(preset_values, template_values, "valueColorBias")
    update_session_value_from_preset(preset_values, template_values, "getAlphaFromValue")
    update_session_value_from_preset(preset_values, template_values, "colorOverflow")
    update_session_value_from_preset(preset_values, template_values, "colorWaveOrigin")
    update_session_value_from_preset(preset_values, template_values, "colorWaveSpeed")
    update_session_value_from_preset(preset_values, template_values, "colorWaveSize")
    update_session_value_from_preset(preset_values, template_values, "colorWaveInwards")
    update_session_value_from_preset(preset_values, template_values, "noiseAmount")
    update_session_value_from_preset(preset_values, template_values, "noiseSmoothing")
    update_session_value_from_preset(preset_values, template_values, "brightness")
    update_session_value_from_preset(preset_values, template_values, "gamma")
    update_session_value_from_preset(preset_values, template_values, "effectRepeats")
    update_session_value_from_preset(preset_values, template_values, "acceleration")
    update_session_value_from_preset(preset_values, template_values, "particleSize")
    update_session_value_from_preset(preset_values, template_values, "patternSplits")
    update_session_value_from_preset(preset_values, template_values, "patternSpread")
    update_session_value_from_preset(preset_values, template_values, "patternFlip")
    update_session_value_from_preset(preset_values, template_values, "patternCenter")
    update_session_value_from_preset(preset_values, template_values, "patternSectionSizeMod")
    update_session_value_from_preset(preset_values, template_values, "fftSize")
    update_session_value_from_preset(preset_values, template_values, "bpmLimit")
    update_session_value_from_preset(preset_values, template_values, "audioResponseCurve")
    update_session_value_from_preset(preset_values, template_values, "audioPeakHoldTimeMs")
    update_session_value_from_preset(preset_values, template_values, "maxWaves")
    update_session_none_templateable_value_from_preset(preset_values, "nextEffectId")
    update_session_value_from_preset(preset_values, template_values, "effectDurationMs")
    update_session_value_from_preset(preset_values, template_values, "effectTransitionDurationMs")
    update_session_value_from_preset(preset_values, template_values, "effectTransitionWarmupDuration")
    update_session_value_from_preset(preset_values, template_values, "resetEffectAfterTransition")
    update_session_value_from_preset(preset_values, template_values, "fluxInfluence")
    update_session_value_from_preset(preset_values, template_values, "energyInfluence")

    update_session_value_from_preset2(preset_values, template_values, "audioMode", "audioModeValue")
    update_session_value_from_preset2(preset_values, template_values, "effectMode", "effectModeValue")
    update_session_value_from_preset2(preset_values, template_values, "colorMode", "colorModeValue")
    st.session_state.audioMode = AUDIO_MODES.get(st.session_state.audioModeValue, AUDIO_MODES[0])
    st.session_state.effectMode = EFFECT_MODES.get(st.session_state.effectModeValue, EFFECT_MODES[0])
    st.session_state.colorMode = COLOR_MODES.get(st.session_state.colorModeValue, COLOR_MODES[0])


def update_session_state_from_static_settings(static_settings_data):
    st.session_state.fps = static_settings_data.get("fps", STATIC_DEFAULTS["fps"])
    st.session_state.ledCount = static_settings_data.get("ledCount", STATIC_DEFAULTS["ledCount"])


def save_presets():
    idx = st.session_state.preset_index
    cur_p = st.session_state.presets[idx]
    templateId = st.session_state.get("templateId")
    template_preset = None
    if templateId is not None:
        template_preset = next((pre for pre in st.session_state.presets if pre["id"] == templateId), None)

    if template_preset is not None:
        template_values = template_preset["values"]
        cur_values = cur_p["values"]
        for k in DYNAMIC_DEFAULTS.keys():
            if k == "templateId" or k == "nextEffectId":
                continue
            if k in cur_values and not st.session_state.get(k + "_ow", False):
                cur_values.pop(k, None)
                tmpl_val = template_values.get(k, DYNAMIC_DEFAULTS.get(k))
                if k == "colors":
                    st.session_state.colors = copy.deepcopy(tmpl_val if tmpl_val is not None else DYNAMIC_DEFAULTS["colors"])
                    st.session_state.colorsRaw = copy.deepcopy(st.session_state.colors)
                    for ci, c in enumerate(st.session_state.colors):
                        st.session_state[f"color_{ci}"] = c["color"]
                        st.session_state[f"threshold_{ci}"] = c["threshold"]
                elif k == "audioMode":
                    st.session_state.audioModeValue = tmpl_val
                    st.session_state.audioMode = AUDIO_MODES.get(tmpl_val, AUDIO_MODES[0])
                elif k == "effectMode":
                    st.session_state.effectModeValue = tmpl_val
                    st.session_state.effectMode = EFFECT_MODES.get(tmpl_val, EFFECT_MODES[0])
                elif k == "colorMode":
                    st.session_state.colorModeValue = tmpl_val
                    st.session_state.colorMode = COLOR_MODES.get(tmpl_val, COLOR_MODES[0])
                else:
                    st.session_state[k] = tmpl_val

    current_values = get_current_preset_values_from_state(False)
    cur_p["values"] = current_values

    if template_preset is not None:
        template_preset["values"] = get_current_preset_values_from_state(True)

    write_presets_to_disk()


def save_static_config():
    current_values = load_static_settings()
    current_values["fps"] = st.session_state.fps
    current_values["ledCount"] = st.session_state.ledCount
    st.session_state.static_settings = current_values

    write_static_config_to_disk(current_values)


def ow_widget(content_lambda, checkbox_key, use_margin=True):
    checkbox_key = checkbox_key + "_ow"
    if st.session_state.get("templateId") is not None:
        if use_margin is True:
            st.markdown(f"""
            <style>
            div[class*="st-key-{checkbox_key}"] {{
                margin-top: 27px;
            }}
            </style>
            """, unsafe_allow_html=True)

        with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
            content_lambda()
            st.checkbox("✏️", key=checkbox_key, on_change=save_presets)
    else:
        content_lambda()
