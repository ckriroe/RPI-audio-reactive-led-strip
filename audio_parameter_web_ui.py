import streamlit as st
import streamlit.components.v1 as components
import json
import os
import copy
import uuid

STATIC_CONFIG_FILE = "static_settings.json"
PRESETS_FILE = "presets.json"

AUDIO_MODES = {
    0: "Gleitender Mittelwert",
    1: "Satisch",
    2: "Transientenerkennung"
}

EFFECT_MODES = {
    0: "Pulsieren",
    1: "Linie",
    2: "Welle",
    3: "Spektrum",
    4: "Partikel",
    5: "Statisch (Aufsteigend)",
    6: "Statisch (Wert von 1.0)",
    7: "Extern",
    8: "Linienverlauf"
}

COLOR_MODES = {
    0: "Wert",
    1: "Index",
    2: "Distanz zur Mitte",
    3: "Distanz zum Rand",
    4: "Farbwelle",
    5: "Farbinseln"
}

DYNAMIC_DEFAULTS = {
    "colors": [
        { "color": "#000000", "threshold": 0.25 },
        { "color": "#FFFFFF", "threshold": 1.0 }
    ],
    "useRainbow": False,
    "effectOrigin": 150,
    "speed": 100,
    "minFreq": 0,
    "maxFreq": 180,
    "fade": 0.6,
    "fadeOverTime": 0.0,
    "bouncyWave": False,
    "staticSpectrum": False,
    "spectrumSections": 10,
    "saturate": 0.6,
    "saturateThreshold": 0.3,
    "meanValueBufferSize": 20,
    "meanValueThreshold": 0.3,
    "audioMode": 0,
    "effectMode": 0,
    "colorMode": 0,
    "minFreqAmplitude": 0.1,
    "maxFreqAmplitude": 10.0,
    "colorIncreaseFactor": 1.0,
    "valueIncreaseFactor": 1.0,
    "colorTransition": 0.25,
    "valueColorBias": 0.0,
    "getAlphaFromValue": False,
    "colorOverflow": False,
    "colorWaveOrigin": 150,
    "colorWaveSpeed": 50,
    "colorWaveSize": 100,
    "colorWaveInwards": False,
    "noiseAmount": 0.00,
    "noiseSmoothing": 1.00,
    "brightness": 1.00,
    "gamma": 1.00,
    "effectRepeats": 1,
    "acceleration": 1.0,
    "particleSize": 20,
    "patternSplits": 0,
    "patternSpread": 0,
    "patternFlip": -1,
    "patternCenter": 0,
    "patternSectionSizeMod": 0.0,
    "fftSize": 512,
    "bpmLimit": -1,
    "audioResponseCurve": 1.0,
    "audioPeakHoldTimeMs": 0,
    "nextEffectId": None,
    "effectDurationMs": 5000,
    "effectTransitionDurationMs": 2500,
    "effectTransitionWarmupDuration": 1000,
    "resetEffectAfterTransition": False,
    "fluxInfluence": 1.0,
    "energyInfluence": 0.5,
    "templateId": None
}

STATIC_DEFAULTS = {
    "sampleRate": 48000,
    "channels": 2,
    "audioDeviceId": 0,
    "belowMinFreqAmplitudeFunctionFactor": -0.03,
    "maxFreqAmplitudeIncreaseRatio": 0.22, # Higher = rise faster
    "maxFreqAmplitudeDecreaseRatio": 0.5, # Higher = fall faster
    "maxFreqAmplitudeValueMultiplier": 0.95,
    "maxFreqAmplitudeTTL": 4000,
    "maxFreqAmplitudeProlongerThreshholdPercent": 0.03,
    "maxFreqAmplitudeDecayRate": 0.002,
    "ledUpdateFrequency": 800000,
    "bounceLayers": 1,
    "maxEffectSpeed": 600,
    "externalModeRelayGpio": 5,
    "invalidFrameSleepTime": 2000,
    "printFrameTimes": False,
    "printFrequencyInfos": False,
    "printSequenceInfos": False,
    "minSanitizedValue": 0.01,
    "useGuiVisualization": True,
    "useLedVisualization": True,
    "guiWidth": 800,
    "guiHeight": 200,
    "mainThreadSettingsCheckIntervalMs": 2000,
    "runIndefinitely": True,
    "outputWhenGpioOff": False,
    "accurateSleeping": False,
    "guiVisualizationMode": 0,
    "rectangleGuiVisualizationHeight": 50,
    "fps": 60,
    "ledCount": 300,
    "fastFluxSmoothing": 1.0,
    "fastEnergySmoothing": 1.0,
    "slowFluxSmoothing": 0.01,
    "slowEnergySmoothing": 0.01
}

NONE_OPTION = "__none__"

def load_presets():
    if os.path.exists(PRESETS_FILE):
        print("Loading preset file...\n")
        with open(PRESETS_FILE, "r") as f:
            jsn = json.load(f)
            print("Preset file loaded:\n", jsn, "\n")
            return jsn
    else:
        print("Preset file did not yet exist, creating presets file\n")
        return create_default_presets_file()

def load_static_settings():
    if os.path.exists(STATIC_CONFIG_FILE):
        print("Loading static config file...\n")
        with open(STATIC_CONFIG_FILE, "r") as f:
             jsn = json.load(f)
             print("Static config file loaded:\n", jsn, "\n")
             return jsn
    else:
        print("Static config file did not yet exist, creating static config file\n")
        return create_default_static_settings_file()

def create_default_presets_file():
    preset_id = str(uuid.uuid4())
    initial_data = {
        "selectedPresetId": preset_id,
        "presets": [
            {
                "name": "Standard",
                "id": preset_id,
                "values": copy.deepcopy(DYNAMIC_DEFAULTS)
            }
        ]
    }
    with open(PRESETS_FILE, "w") as f:
        json.dump(initial_data, f, indent=2)

    return initial_data

def create_default_static_settings_file():
    initial_data = copy.deepcopy(STATIC_DEFAULTS)
    write_static_config_to_disk(initial_data)
    return initial_data

def write_presets_to_disk():
    data = {
        "selectedPresetId": st.session_state.preset_guid,
        "presets": st.session_state.presets
    }
    print("Writing presets to disk:\n", data, "\n")
    with open(PRESETS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def write_static_config_to_disk(values):
    print("Writing static config to disk:\n", values, "\n")
    with open(STATIC_CONFIG_FILE, "w") as f:
        json.dump(values, f, indent=2)

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
    get_for_template = get_for_template and template_values != None
    if st.session_state.get("templateId") != None:
        if st.session_state.get(preset_key + "_ow") == True:
            if get_for_template == True:
                stateObj[preset_key] = template_values[preset_key]
            else:
                stateObj[preset_key] = st.session_state[session_key]
        elif get_for_template == True:
            stateObj[preset_key] = st.session_state[session_key]
    else:
        stateObj[preset_key] = st.session_state[session_key]

def set_value_from_state(stateObj, get_for_template, template_values, key):
    set_value_from_state2(stateObj, get_for_template, template_values, key, key)

def set_none_templateable_value_from_state(stateObj, get_for_template, template_values, key):
    if get_for_template == True:
        if template_values != None:
            stateObj[key] = template_values[key]
    else:
        stateObj[key] = st.session_state[key]

def get_current_preset_values_from_state(get_for_template):
    sync_widgets_to_colors_list()
   
    st.session_state.audioModeValue = next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode) 
    st.session_state.effectModeValue = next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
    st.session_state.colorModeValue = next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)
    
    template_values = None
    templateId = st.session_state.get("templateId")
    if templateId != None:
        template_values = next((pre["values"] for pre in st.session_state.presets if pre["id"] == templateId), None)

    stateObj = {
        "templateId": st.session_state.templateId
    }

    if get_for_template == True:
        stateObj["templateId"] = None
        
    set_value_from_state(stateObj, get_for_template, template_values, "colors")
    set_value_from_state(stateObj, get_for_template, template_values, "useRainbow")
    set_value_from_state(stateObj, get_for_template, template_values, "effectOrigin")
    set_value_from_state(stateObj, get_for_template, template_values, "speed")
    set_value_from_state(stateObj, get_for_template, template_values, "minFreq")
    set_value_from_state(stateObj, get_for_template, template_values, "maxFreq")
    set_value_from_state(stateObj, get_for_template, template_values, "fade")
    set_value_from_state(stateObj, get_for_template, template_values, "fadeOverTime")
    set_value_from_state(stateObj, get_for_template, template_values, "staticSpectrum")
    set_value_from_state(stateObj, get_for_template, template_values, "spectrumSections")
    set_value_from_state(stateObj, get_for_template, template_values, "bouncyWave")
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
    if template_values != None:
        if preset_key in preset_values:
            st.session_state[preset_key + "_ow"] = True
        else:
            preset_value = template_values.get(preset_key)
            st.session_state[preset_key + "_ow"] = False

    if preset_value == None:
        preset_value = DYNAMIC_DEFAULTS[preset_key]

    st.session_state[session_key] = preset_value

def update_session_value_from_preset(preset_values, template_values, key):
    update_session_value_from_preset2(preset_values, template_values, key, key)

def update_session_none_templateable_value_from_preset(preset_values, key):
    preset_value = preset_values.get(key)
    if preset_value == None:
        preset_value = DYNAMIC_DEFAULTS[key]

    st.session_state[key] = preset_value

def update_session_state_from_preset(preset_data):
    st.session_state.preset_guid = preset_data["id"]
    preset_values = preset_data["values"]

    template_values = None
    templateId = preset_values.get("templateId")
    if templateId != None:
        template_values = next((pre["values"] for pre in st.session_state.presets if pre["id"] == templateId), None)

    st.session_state.rename_input = preset_data["name"]
    st.session_state.templateId = preset_values.get("templateId", DYNAMIC_DEFAULTS["templateId"])

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
    current_values = get_current_preset_values_from_state(False)
    
    idx = st.session_state.preset_index
    st.session_state.presets[idx]["values"] = current_values
    
    templateId = st.session_state.get("templateId")
    if templateId != None:
        templateIndex = next((i for i, pre in enumerate(st.session_state.presets) if pre["id"] == templateId), None)
        if templateIndex != None:
            st.session_state.presets[templateIndex]["values"] = get_current_preset_values_from_state(True)

    write_presets_to_disk()

def save_static_config():
    current_values = load_static_settings()
    current_values["fps"] = st.session_state.fps
    current_values["ledCount"] = st.session_state.ledCount
    st.session_state.static_settings = current_values

    write_static_config_to_disk(current_values)

def cb_create_preset():
    curr_index = st.session_state.preset_index
    new_index = curr_index + 1
    new_id = str(uuid.uuid4())
    current_data = copy.deepcopy(st.session_state.presets[curr_index])
    current_data["name"] = f"{current_data['name']} (Kopieren)"
    current_data["id"] = new_id
    st.session_state.presets.insert(new_index, current_data)
 
    st.session_state.preset_guid = new_id
    st.session_state.preset_index = new_index
    update_session_state_from_preset(st.session_state.presets[new_index])
    
    write_presets_to_disk()

def cb_move_preset_back(): 
    curr_index = st.session_state.preset_index
    new_index = curr_index - 1
    tmp = st.session_state.presets[new_index]
    st.session_state.presets[new_index] = st.session_state.presets[curr_index]
    st.session_state.presets[curr_index] = tmp
    st.session_state.preset_index = new_index
    update_session_state_from_preset(st.session_state.presets[new_index])
    write_presets_to_disk()

def cb_move_preset_forward(): 
    curr_index = st.session_state.preset_index
    new_index = curr_index + 1
    tmp = st.session_state.presets[new_index]
    st.session_state.presets[new_index] = st.session_state.presets[curr_index]
    st.session_state.presets[curr_index] = tmp
    st.session_state.preset_index = new_index
    update_session_state_from_preset(st.session_state.presets[new_index])
    write_presets_to_disk()

def cb_delete_preset():
    if len(st.session_state.presets) > 1 and st.session_state.preset_index != 0:
        removed_preset = st.session_state.presets.pop(st.session_state.preset_index)
        if removed_preset is None:
            return

        for i in range(len(st.session_state.presets)):
            preVals = st.session_state.presets[i]["values"]
            if preVals.get("nextEffectId") == removed_preset["id"]:
                preVals["nextEffectId"] = None

            if preVals.get("templateId") == removed_preset["id"]:
                apply_settings_from_template(removed_preset["values"], preVals)

        st.session_state.preset_index = max(0, st.session_state.preset_index - 1)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        write_presets_to_disk()

def cb_add_color():
    sync_widgets_to_colors_list()
    st.session_state.colors.append({ "color": "#ffffff", "threshold": 1.0 })
    
    new_idx = len(st.session_state.colors) - 1
    st.session_state[f"color_{new_idx}"] = "#ffffff"
    st.session_state[f"threshold_{new_idx}"] = 1.0
    
    save_presets()

def remove_color(index):
    if len(st.session_state.colors) <= 2:
        return

    st.session_state.colors.pop(index)
    for i in range(index, len(st.session_state.colors) + 1):
        if f"color_{i + 1}" in st.session_state:
            st.session_state[f"color_{i}"] = st.session_state[f"color_{i + 1}"]

        if f"threshold_{i + 1}" in st.session_state:
            st.session_state[f"threshold_{i}"] = st.session_state[f"threshold_{i + 1}"]

    last_idx = len(st.session_state.colors)
    st.session_state.pop(f"color_{last_idx}", None)
    st.session_state.pop(f"threshold_{last_idx}", None)
    sync_widgets_to_colors_list()

    save_presets()

st.set_page_config(page_title="RPISC Einstellungen", page_icon="🎛️", layout="centered")
preset_data = load_presets()
st.session_state.presets = preset_data["presets"]
st.session_state.preset_guid = preset_data["selectedPresetId"]

preset_index = 0
for i in range(len(st.session_state.presets)):
    if st.session_state.presets[i]["id"] == st.session_state.preset_guid:
        preset_index = i
        break

st.session_state.preset_index = preset_index    
if st.session_state.preset_index >= len(st.session_state.presets):
    st.session_state.preset_index = 0
    st.session_state.preset_guid = st.session_state.presets[0]["id"]
    
if "rename_input" not in st.session_state:
    st.session_state.rename_input = st.session_state.presets[st.session_state.preset_index]["name"]

update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])

if "static_settings" not in st.session_state:
     st.session_state.static_settings = load_static_settings()
     update_session_state_from_static_settings(st.session_state.static_settings)

slider_fix_js = """
<script>
(function() {
    const parentWindow = window.parent;
    const parentDoc = parentWindow.document;

    function isMobile() {
        const hasTouch = 'ontouchstart' in parentWindow || parentWindow.navigator.maxTouchPoints > 0;
        const isSmallScreen = parentWindow.innerWidth <= 768;
        return hasTouch && isSmallScreen;
    }

    function interceptSliderEvents() {
        const sliders = parentDoc.querySelectorAll('[data-baseweb="slider"]');
        
        sliders.forEach((slider, index) => {
            if (slider.getAttribute('data-slider-scroll-fixed') === 'true') return;
            const eventsToIntercept = ['pointerdown', 'touchstart', 'mousedown', 'pointerup', 'touchend', 'mouseup', 'click'];
            
            eventsToIntercept.forEach(eventType => {
                slider.addEventListener(eventType, function(e) {
                    if (!isMobile()) return;
                    const thumb = slider.querySelector('[role="slider"]');

                    if (thumb && e.target !== thumb && !thumb.contains(e.target)) {
                        e.stopPropagation(); 
                    }
                }, { capture: true, passive: true });
            });

            slider.setAttribute('data-slider-scroll-fixed', 'true');
        });
    }

    const observer = new MutationObserver((mutations) => {
        interceptSliderEvents();
    });

    observer.observe(parentDoc.body, { childList: true, subtree: true });
    interceptSliderEvents();
})();
</script>
"""

components.html(slider_fix_js, height=0, width=0)

st.title("RPISC Einstellungen")
st.info(f"Aktive Voreinstellung: **{st.session_state.presets[st.session_state.preset_index]['name']}**")

guid_to_obj = {v["id"]: v for v in st.session_state.presets}
current_guid = st.session_state.preset_guid

with st.expander("📂 Voreinstellungs Verwaltung", expanded=False):
    preset_names = [p["name"] for p in st.session_state.presets]
    
    def on_preset_change():
        new_index = st.session_state.preset_selector_idx
        st.session_state.preset_index = new_index
        update_session_state_from_preset(st.session_state.presets[new_index])
        write_presets_to_disk()

    st.selectbox(
        "Voreinstellung auswählen",
        range(len(preset_names)),
        format_func=lambda x: preset_names[x],
        index=st.session_state.preset_index,
        key="preset_selector_idx",
        on_change=on_preset_change
    )

    def on_rename():
        new_name = st.session_state.rename_input
        if new_name:
            st.session_state.presets[st.session_state.preset_index]["name"] = new_name
            write_presets_to_disk()

    st.text_input(
        "Voreinstellung umbenennen",
        key="rename_input",
        disabled=(st.session_state.preset_index == 0),
        on_change=on_rename
    )

    col_move_up, col_move_down = st.columns([1, 1])

    with col_move_up:
        st.button("▲ Nach oben", disabled=(st.session_state.preset_index == 0 or st.session_state.preset_index == 1), on_click=cb_move_preset_back, width="stretch")
        st.button("▼ Nach unten", disabled=(st.session_state.preset_index == 0 or st.session_state.preset_index == len(st.session_state.presets) - 1), on_click=cb_move_preset_forward, width="stretch")

    with col_move_down:
        st.button("➕ Neu (Kopieren)", on_click=cb_create_preset, width="stretch")
        st.button("🗑️ Löschen", disabled=(st.session_state.preset_index == 0), on_click=cb_delete_preset, width="stretch")

    is_master = any(pre["values"].get("templateId") == current_guid for pre in st.session_state.presets)
    if st.session_state.preset_index != 0 and is_master == False:
        filtered_values = [v for v in st.session_state.presets if v["id"] != st.session_state.preset_guid and v["values"].get("templateId") is None]
        options = [NONE_OPTION] + [v["id"] for v in filtered_values]
        if "templateId" not in st.session_state:
            st.session_state.templateId = None

        template_guid = st.session_state.templateId

        default_index = 0
        if template_guid is not None:
            default_index = next(
                (i for i, v in enumerate(options) if v is not NONE_OPTION and v == template_guid),
                0
            )

        def on_template_changed():
            selected_guid = st.session_state["presetEntry"]
            if selected_guid == NONE_OPTION:
                st.session_state.templateId = None
            else:
                st.session_state.templateId = selected_guid
        
            save_presets()

        st.selectbox(
            "Vorlage",
            options,
            index=default_index,
            format_func=lambda x: "Deaktiviert" if x is NONE_OPTION else guid_to_obj[x]["name"],
            key="presetEntry",
            on_change=on_template_changed
        )

def get_audio_mode_id():
    return next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode)
def get_effect_mode_id():
    return next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
def get_color_mode_id():
    return next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)

# Overwrite widget func
def ow_widget(content_lambda, checkbox_key, use_margin=True):
    checkbox_key = checkbox_key + "_ow"
    if st.session_state.get("templateId") != None:
        if use_margin == True:
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

with st.expander("Audio", expanded=False):
    ow_widget(lambda: st.selectbox("Audiomodus", list(AUDIO_MODES.values()), key="audioMode", on_change=save_presets), "audioMode")
    ow_widget(lambda: st.slider("Min. Lautstärke", 0.0, 50.0, step=0.01, key="minFreqAmplitude", on_change=save_presets), "minFreqAmplitude")
    ow_widget(lambda: st.slider("Max. Lautstärke", 0.0, 50.0, step=0.01, key="maxFreqAmplitude", on_change=save_presets), "maxFreqAmplitude")

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

with st.expander("Effekt", expanded=False):
    ow_widget(lambda: st.selectbox("Effektmodus", list(EFFECT_MODES.values()), key="effectMode", on_change=save_presets), "effectMode")
    ow_widget(lambda: st.number_input("Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="effectOrigin", on_change=save_presets), "effectOrigin")
    ow_widget(lambda: st.slider("Effekt Beschleunigung", 0.0, 5.0, step=0.01, key="acceleration", on_change=save_presets), "acceleration")

    if get_effect_mode_id() in (2, 4):
        ow_widget(lambda: st.slider("Effekt Geschwindigkeit", 1, 600, key="speed", on_change=save_presets), "speed")

    ow_widget(lambda: st.slider("Effekt Verstärkung", 0.1, 10.0, key="valueIncreaseFactor", on_change=save_presets), "valueIncreaseFactor")

    if get_effect_mode_id() == 3:
        ow_widget(lambda: st.toggle("Statisches Spektrum", key="staticSpectrum", on_change=save_presets), "staticSpectrum", False)
        ow_widget(lambda: st.slider("Spektrum Sektionen", 1, 50, step=1, key="spectrumSections", on_change=save_presets), "spectrumSections")

    if get_effect_mode_id() == 4:
        ow_widget(lambda: st.slider("Partikel Größe", 1, 1000, step=1, key="particleSize", on_change=save_presets), "particleSize")

    if get_effect_mode_id() == 2:
        ow_widget(lambda: st.toggle("Abprallen", key="bouncyWave", on_change=save_presets), "bouncyWave", False)

    if get_effect_mode_id() == 2 or get_effect_mode_id() == 8:
        ow_widget(lambda: st.slider("Verblassung nach Zeit", -0.999, 0.999, step=0.001, key="fadeOverTime", on_change=save_presets), "fadeOverTime")

    ow_widget(lambda: st.slider("Verblassung", 0.001, 0.999, step=0.001, key="fade", on_change=save_presets), "fade")
    ow_widget(lambda: st.slider("Sättigung", 0.01, 1.0, step=0.01, key="saturate", on_change=save_presets), "saturate")
    ow_widget(lambda: st.slider("Sättigungs Grenzwert", 0.0, 1.0, step=0.01, key="saturateThreshold", on_change=save_presets), "saturateThreshold")

with st.expander("Muster", expanded=False):
    ow_widget(lambda: st.slider("Effekt Wiederholungen", 1, 50, step=1, key="effectRepeats", on_change=save_presets), "effectRepeats")
    ow_widget(lambda: st.slider("Muster Teilungen", 0, 50, step=1, key="patternSplits", on_change=save_presets), "patternSplits")
    ow_widget(lambda: st.slider("Jedes N'te Muster verdrehen", -1, 50, step=1, key="patternFlip", on_change=save_presets), "patternFlip")
    ow_widget(lambda: st.number_input("Muster Verteilung", min_value=0, max_value=999999, step=1, format="%d", key="patternSpread", on_change=save_presets), "patternSpread")
    ow_widget(lambda: st.number_input("Muster Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="patternCenter", on_change=save_presets), "patternCenter")
    ow_widget(lambda: st.slider("Muster Größenveränderung", -5.00, 5.00, step=0.01, key="patternSectionSizeMod", on_change=save_presets), "patternSectionSizeMod")

with st.expander("Sequenzierung", expanded=False):
    filtered_values = [v for v in st.session_state.presets if v["id"] != st.session_state.preset_guid]
    options = [NONE_OPTION] + [v["id"] for v in filtered_values]
    if "nextEffectId" not in st.session_state:
        st.session_state.nextEffectId = None

    linked_guid = st.session_state.nextEffectId

    default_index = 0
    if linked_guid is not None:
        default_index = next(
            (i for i, v in enumerate(options) if v is not NONE_OPTION and v == linked_guid),
            0
        )

    def on_next_effect_changed():
        selected_guid = st.session_state["nextEffectEntry"]
        if selected_guid == NONE_OPTION:
            st.session_state.nextEffectId = None
        else:
            st.session_state.nextEffectId = selected_guid
        
        save_presets()

    st.selectbox(
        "Nächster Effekt",
        options,
        index=default_index,
        format_func=lambda x: "Deaktiviert" if x is NONE_OPTION else guid_to_obj[x]["name"],
        key="nextEffectEntry",
        on_change=on_next_effect_changed
    )

    ow_widget(lambda: st.number_input("Effekt Dauer in ms", min_value=1, max_value=99999999, step=1, format="%d", key="effectDurationMs", on_change=save_presets), "effectDurationMs")
    ow_widget(lambda: st.number_input("Effekt Übergangsdauer in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionDurationMs", on_change=save_presets), "effectTransitionDurationMs")
    ow_widget(lambda: st.number_input("Übergangs Vorlaufzeit in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionWarmupDuration", on_change=save_presets), "effectTransitionWarmupDuration")
    ow_widget(lambda: st.toggle("Effekt nach Übergang zurücksetzen", key="resetEffectAfterTransition", on_change=save_presets), "resetEffectAfterTransition", False)

with st.expander("Farben", expanded=False):
    ow_widget(lambda: st.selectbox("Farbmodus", list(COLOR_MODES.values()), key="colorMode", on_change=save_presets), "colorMode")
    ow_widget(lambda: st.slider("Helligkeit", 0.0, 5.0, step=0.01, key="brightness", on_change=save_presets), "brightness")
    ow_widget(lambda: st.slider("Gamma", 0.0, 5.0, step=0.01, key="gamma", on_change=save_presets), "gamma")
    ow_widget(lambda: st.slider("Farbverstärkung", 0.1, 20.0, key="colorIncreaseFactor", on_change=save_presets), "colorIncreaseFactor")
    ow_widget(lambda: st.slider("Farbverlauf", 0.00, 0.50, key="colorTransition", on_change=save_presets), "colorTransition")

    if get_color_mode_id() != 0:
        ow_widget(lambda: st.slider("Wert / Farb Verschmierung", 0.00, 1.00, step=0.01, key="valueColorBias", on_change=save_presets), "valueColorBias")
        ow_widget(lambda: st.toggle("Transparenz aus Wert", key="getAlphaFromValue", on_change=save_presets), "getAlphaFromValue", False)

    if get_color_mode_id() == 4:
        ow_widget(lambda: st.number_input("Farbwellen Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="colorWaveOrigin", on_change=save_presets), "colorWaveOrigin")
        ow_widget(lambda: st.number_input("Farbwellen Größe", min_value=1, max_value=999999, step=1, format="%d", key="colorWaveSize", on_change=save_presets), "colorWaveSize")
        ow_widget(lambda: st.slider("Farbwellen Geschwindigkeit", 1, 600, key="colorWaveSpeed", on_change=save_presets), "colorWaveSpeed")
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
        

        col_color, col_threshold = st.columns([1, 2])
        with col_color:
            with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
                st.color_picker(
                    label,
                    key=f"color_{i}",
                    on_change=save_presets
                )
        with col_threshold:
            if i >= len(st.session_state.colors) - 1:
                st.session_state[f"threshold_{i}"] = 1.0

            with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
                st.slider(
                    "Grenzwert",
                    min_value=0.00,
                    max_value=1.00,
                    step=0.01,
                    key=f"threshold_{i}",
                    disabled=(i >= len(st.session_state.colors) - 1),
                    label_visibility="visible",
                    on_change=save_presets
                )
                if i > 1:
                    st.button("❌", disabled=(len(st.session_state.colors) <= 2), key=f"rm_col_{i}", on_click=remove_color, args=(i,))

with st.expander("Globale Einstellungen", expanded=False):
    st.number_input("FPS", min_value=1, max_value=400, step=1, format="%d", key="fps", on_change=save_static_config)
    st.number_input("LED Anzahl", min_value=2, max_value=99999, step=1, format="%d", key="ledCount", on_change=save_static_config)
