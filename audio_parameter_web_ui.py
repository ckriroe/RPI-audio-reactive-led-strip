import streamlit as st
import streamlit.components.v1 as components
import json
import os
import copy
import uuid
import re

STATIC_CONFIG_FILE = "static_settings.json"
PRESETS_FILE = "presets.json"
VERSION_FILE = "version.txt"

def load_version():
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                ver = f.read().strip()
                if ver:
                    return ver
        except Exception:
            pass
    return None

MIN_VOLUME_DB = -120.0
MAX_VOLUME_DB = 0.0

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
    8: "Linienverlauf",
    9: "Wasser Simulation",
    10: "Stroboskop",
    11: "Atmung (Ambient)",
    12: "Feuer Simulation",
    13: "Meteor",
    14: "VU-Meter"
}

COLOR_MODES = {
    0: "Wert",
    1: "Index",
    2: "Distanz zur Mitte",
    3: "Distanz zum Rand",
    4: "Farbwelle",
    5: "Farbinseln",
    6: "Farbzyklus (Leerer Streifen)",
    7: "Farbzyklus (Zeit)",
    8: "Farbzyklus (Audio)"
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
    "minFreqAmplitude": MIN_VOLUME_DB,
    "maxFreqAmplitude": MAX_VOLUME_DB,
    "colorIncreaseFactor": 1.0,
    "valueIncreaseFactor": 1.0,
    "colorTransition": 0.25,
    "colorCycleSpeed": 1.0,
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
    "maxWaves": 4,
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
    "maxEffectSpeed": 1000,
    "externalModeRelayGpio": 5,
    "invalidFrameSleepTime": 2000,
    "printFrameTimes": False,
    "printFrequencyInfos": False,
    "printSequenceInfos": False,
    "printAudioDevices": False,
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
GLOBAL_GROUP_ID = "__global__"

def load_presets():
    if os.path.exists(PRESETS_FILE):
        print("Loading preset file...\n")
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            jsn = json.load(f)
            if "groups" not in jsn:
                jsn["groups"] = []
            for preset in jsn.get("presets", []):
                if "groupId" not in preset:
                    preset["groupId"] = None
                vals = preset.get("values", {})
                if "minFreqAmplitude" in vals and (vals["minFreqAmplitude"] < MIN_VOLUME_DB or vals["minFreqAmplitude"] > MAX_VOLUME_DB):
                    vals["minFreqAmplitude"] = MIN_VOLUME_DB
                if "maxFreqAmplitude" in vals and (vals["maxFreqAmplitude"] < MIN_VOLUME_DB or vals["maxFreqAmplitude"] > MAX_VOLUME_DB):
                    vals["maxFreqAmplitude"] = MAX_VOLUME_DB
            if jsn.get("presets") and len(jsn["presets"]) > 0:
                jsn["presets"][0]["groupId"] = None
            print("Preset file loaded:\n", jsn, "\n")
            return jsn
    else:
        print("Preset file did not yet exist, creating presets file\n")
        return create_default_presets_file()

def load_static_settings():
    if os.path.exists(STATIC_CONFIG_FILE):
        print("Loading static config file...\n")
        with open(STATIC_CONFIG_FILE, "r", encoding="utf-8") as f:
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
        "groups": [],
        "presets": [
            {
                "name": "Standard",
                "id": preset_id,
                "groupId": None,
                "values": copy.deepcopy(DYNAMIC_DEFAULTS)
            }
        ]
    }
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(initial_data, f, indent=2)

    return initial_data

def create_default_static_settings_file():
    initial_data = copy.deepcopy(STATIC_DEFAULTS)
    write_static_config_to_disk(initial_data)
    return initial_data

def write_presets_to_disk():
    data = {
        "selectedPresetId": st.session_state.preset_guid,
        "groups": st.session_state.get("groups", []),
        "presets": st.session_state.presets
    }
    print("Writing presets to disk:\n", data, "\n")
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_static_config_to_disk(values):
    print("Writing static config to disk:\n", values, "\n")
    with open(STATIC_CONFIG_FILE, "w", encoding="utf-8") as f:
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


def extract_base_name(name):
    m = re.match(r"^(.*?)\s*\(Kopieren(?:\s+\d+)?\)$", name.strip())
    return m.group(1).strip() if m else name.strip()


def is_preset_template_master(preset_id):
    if not st.session_state.presets or preset_id == st.session_state.presets[0]["id"]:
        return False
    return any(p["values"].get("templateId") == preset_id for p in st.session_state.presets)


def get_preset_icon_tag(preset):
    if is_preset_template_master(preset["id"]):
        return " ⭐"
    if preset["values"].get("templateId") is not None:
        return " 🔗"
    return ""


def get_group_name(group_id):
    if group_id == GLOBAL_GROUP_ID or group_id is None:
        return "Global"
    g = next((x for x in st.session_state.groups if x["id"] == group_id), None)
    return g["name"] if g else "Global"


def get_presets_in_group(group_id):
    if group_id == GLOBAL_GROUP_ID or group_id is None:
        return [p for p in st.session_state.presets if p.get("groupId") is None or p.get("groupId") == GLOBAL_GROUP_ID]
    return [p for p in st.session_state.presets if p.get("groupId") == group_id]


def generate_copy_name(original_name, existing_names):
    base_name = extract_base_name(original_name)

    existing_nums = set()
    for name in existing_names:
        if name == f"{base_name} (Kopieren)":
            existing_nums.add(1)
        else:
            m_ex = re.match(rf"^{re.escape(base_name)}\s*\(Kopieren\s+(\d+)\)$", name)
            if m_ex:
                existing_nums.add(int(m_ex.group(1)))

    if not existing_nums or 1 not in existing_nums:
        return f"{base_name} (Kopieren)"
    else:
        next_num = 2
        while next_num in existing_nums:
            next_num += 1
        return f"{base_name} (Kopieren {next_num})"


def select_preset_by_id(preset_id):
    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is not None:
        st.session_state.preset_index = idx
        st.session_state.preset_guid = preset_id
        gid = st.session_state.presets[idx].get("groupId")
        st.session_state.active_group_id = gid if gid else GLOBAL_GROUP_ID
        update_session_state_from_preset(st.session_state.presets[idx])
        write_presets_to_disk()


def copy_preset_by_id(preset_id, new_name=None):
    if preset_id == st.session_state.preset_guid:
        save_presets()

    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is None:
        return

    new_id = str(uuid.uuid4())
    copied_data = copy.deepcopy(st.session_state.presets[idx])
    copied_data["id"] = new_id

    existing_names = [p["name"] for p in st.session_state.presets]
    if new_name and new_name.strip():
        copied_data["name"] = new_name.strip()
    else:
        copied_data["name"] = generate_copy_name(copied_data["name"], existing_names)

    base_root = extract_base_name(st.session_state.presets[idx]["name"])

    insert_idx = idx + 1
    while insert_idx < len(st.session_state.presets):
        next_p = st.session_state.presets[insert_idx]
        if next_p.get("groupId") == copied_data.get("groupId"):
            m_next = re.match(rf"^{re.escape(base_root)}\s*\(Kopieren(?:\s+\d+)?\)$", next_p["name"])
            if m_next:
                insert_idx += 1
                continue
        break

    st.session_state.presets.insert(insert_idx, copied_data)
    select_preset_by_id(new_id)
    st.rerun()


def move_preset_in_group(preset_id, direction):
    preset = next((p for p in st.session_state.presets if p["id"] == preset_id), None)
    if not preset or preset["id"] == st.session_state.presets[0]["id"]:
        return

    group_id = preset.get("groupId") or GLOBAL_GROUP_ID
    siblings = get_presets_in_group(group_id)
    s_idx = next((i for i, p in enumerate(siblings) if p["id"] == preset_id), None)
    if s_idx is None:
        return

    target_s_idx = s_idx + direction
    if group_id == GLOBAL_GROUP_ID and target_s_idx == 0:
        return

    if 0 <= target_s_idx < len(siblings):
        target_preset = siblings[target_s_idx]
        p_idx = st.session_state.presets.index(preset)
        t_idx = st.session_state.presets.index(target_preset)
        st.session_state.presets[p_idx], st.session_state.presets[t_idx] = (
            st.session_state.presets[t_idx],
            st.session_state.presets[p_idx],
        )
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        write_presets_to_disk()


def unassign_preset(preset_id):
    preset = next((p for p in st.session_state.presets if p["id"] == preset_id), None)
    if not preset or preset["id"] == st.session_state.presets[0]["id"]:
        return
    cur_group = preset.get("groupId")
    if not cur_group or cur_group == GLOBAL_GROUP_ID:
        return

    cur_group_presets = get_presets_in_group(cur_group)
    prev_idx_in_group = next((i for i, p in enumerate(cur_group_presets) if p["id"] == preset_id), 0) - 1

    preset["groupId"] = None

    if st.session_state.preset_guid == preset_id:
        remaining = get_presets_in_group(cur_group)
        if len(remaining) > 0:
            target_idx = max(0, prev_idx_in_group)
            if target_idx >= len(remaining):
                target_idx = len(remaining) - 1
            select_preset_by_id(remaining[target_idx]["id"])
        else:
            select_preset_by_id(st.session_state.presets[0]["id"])
            st.session_state.active_group_id = cur_group
    else:
        write_presets_to_disk()


def delete_preset_by_id(preset_id):
    if len(st.session_state.presets) <= 1 or preset_id == st.session_state.presets[0]["id"]:
        return

    idx = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == preset_id), None)
    if idx is None:
        return

    cur_group_presets = get_presets_in_group(st.session_state.active_group_id)
    prev_idx_in_group = next((i for i, p in enumerate(cur_group_presets) if p["id"] == preset_id), 0) - 1

    removed_preset = st.session_state.presets.pop(idx)

    for pre in st.session_state.presets:
        preVals = pre["values"]
        if preVals.get("nextEffectId") == removed_preset["id"]:
            preVals["nextEffectId"] = None
        if preVals.get("templateId") == removed_preset["id"]:
            apply_settings_from_template(removed_preset["values"], preVals)

    if st.session_state.preset_guid == preset_id:
        remaining_in_group = get_presets_in_group(st.session_state.active_group_id)
        if len(remaining_in_group) > 0:
            target_idx = max(0, prev_idx_in_group)
            if target_idx >= len(remaining_in_group):
                target_idx = len(remaining_in_group) - 1
            select_preset_by_id(remaining_in_group[target_idx]["id"])
        else:
            cur_group = st.session_state.active_group_id
            select_preset_by_id(st.session_state.presets[0]["id"])
            st.session_state.active_group_id = cur_group
    else:
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        write_presets_to_disk()


def move_group(group_id, direction):
    idx = next((i for i, g in enumerate(st.session_state.groups) if g["id"] == group_id), None)
    if idx is None:
        return
    new_idx = idx + direction
    if 0 <= new_idx < len(st.session_state.groups):
        st.session_state.groups[idx], st.session_state.groups[new_idx] = (
            st.session_state.groups[new_idx],
            st.session_state.groups[idx],
        )
        st.session_state.active_group_id = group_id
        st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1
        write_presets_to_disk()


def delete_group_by_id(group_id):
    to_delete = [p for p in st.session_state.presets if p.get("groupId") == group_id]
    to_delete_ids = {p["id"] for p in to_delete}
    if len(st.session_state.presets) > 0:
        to_delete_ids.discard(st.session_state.presets[0]["id"])

    for pre in st.session_state.presets:
        if pre["id"] not in to_delete_ids:
            preVals = pre["values"]
            if preVals.get("nextEffectId") in to_delete_ids:
                preVals["nextEffectId"] = None
            if preVals.get("templateId") in to_delete_ids:
                tmpl = next((p for p in to_delete if p["id"] == preVals.get("templateId")), None)
                if tmpl:
                    apply_settings_from_template(tmpl["values"], preVals)
                else:
                    preVals["templateId"] = None

    current_options = [GLOBAL_GROUP_ID] + [g["id"] for g in st.session_state.groups]
    prev_group_id = GLOBAL_GROUP_ID
    if group_id in current_options:
        curr_idx = current_options.index(group_id)
        prev_group_id = current_options[max(0, curr_idx - 1)]

    st.session_state.presets = [p for p in st.session_state.presets if p["id"] not in to_delete_ids]
    st.session_state.groups = [g for g in st.session_state.groups if g["id"] != group_id]
    st.session_state.active_group_id = prev_group_id
    st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1

    presets_in_target = get_presets_in_group(prev_group_id)
    if st.session_state.preset_guid in to_delete_ids or st.session_state.preset_index >= len(st.session_state.presets):
        if len(presets_in_target) > 0:
            select_preset_by_id(presets_in_target[0]["id"])
        else:
            st.session_state.preset_index = 0
            st.session_state.preset_guid = st.session_state.presets[0]["id"]
            update_session_state_from_preset(st.session_state.presets[0])
            write_presets_to_disk()
    else:
        cur_id = st.session_state.preset_guid
        st.session_state.preset_index = next((i for i, p in enumerate(st.session_state.presets) if p["id"] == cur_id), 0)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        write_presets_to_disk()

def get_preset_name(preset_id):
    p = next((x for x in st.session_state.presets if x["id"] == preset_id), None)
    return p["name"] if p else str(preset_id)

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

def move_color(index, direction):
    new_index = index + direction
    if index <= 0 or new_index <= 0 or index >= len(st.session_state.colors) or new_index >= len(st.session_state.colors):
        return

    sync_widgets_to_colors_list()
    st.session_state.colors[index]["color"], st.session_state.colors[new_index]["color"] = (
        st.session_state.colors[new_index]["color"],
        st.session_state.colors[index]["color"],
    )
    st.session_state[f"color_{index}"] = st.session_state.colors[index]["color"]
    st.session_state[f"color_{new_index}"] = st.session_state.colors[new_index]["color"]
    save_presets()

def on_dialog_dismiss():
    st.session_state.active_dialog = None
    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
    st.session_state.pop("dlg_create_grp_name", None)

@st.dialog("Neue Gruppe erstellen", on_dismiss=on_dialog_dismiss)
def dialog_create_group():
    new_name = st.text_input("Gruppenname", key="dlg_create_grp_name")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_create_grp_cancel", use_container_width=True):
            st.session_state.pop("dlg_create_grp_name", None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.button("Erstellen", key="dlg_create_grp_submit", type="primary", use_container_width=True):
            name_clean = new_name.strip() if new_name else ""
            if not name_clean:
                name_clean = "Gruppe"
            new_gid = str(uuid.uuid4())
            st.session_state.groups.append({"id": new_gid, "name": name_clean})
            st.session_state.active_group_id = new_gid
            st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1
            write_presets_to_disk()
            st.session_state.pop("dlg_create_grp_name", None)
            st.session_state.active_dialog = None
            st.rerun()

@st.dialog("Gruppe umbenennen", on_dismiss=on_dialog_dismiss)
def dialog_rename_group(group_id, current_name=None):
    if not current_name:
        current_name = get_group_name(group_id)
    input_key = f"dlg_ren_grp_name_{group_id}"
    new_name = st.text_input("Neuer Gruppenname", value=current_name, key=input_key)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_ren_grp_cancel", use_container_width=True):
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        is_disabled = (len(new_name.strip()) == 0)
        if st.button("Speichern", key="dlg_ren_grp_submit", type="primary", disabled=is_disabled, use_container_width=True):
            name_clean = new_name.strip()
            if name_clean:
                g = next((x for x in st.session_state.groups if x["id"] == group_id), None)
                if g:
                    g["name"] = name_clean
                    write_presets_to_disk()
                    st.session_state.group_selector_ver = st.session_state.get("group_selector_ver", 0) + 1
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            st.rerun()

@st.dialog("Gruppe löschen", on_dismiss=on_dialog_dismiss)
def dialog_delete_group(group_id, group_name=None):
    if not group_name:
        group_name = get_group_name(group_id)
    presets_count = len([p for p in st.session_state.presets if p.get("groupId") == group_id])
    st.write(f"Möchtest du die Gruppe **{group_name}** wirklich löschen?")
    if presets_count > 0:
        st.write(f"⚠️ Es werden auch alle **{presets_count}** Voreinstellung(en) in dieser Gruppe unwiderruflich gelöscht!")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_del_grp_cancel", use_container_width=True):
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.button("Ja, Gruppe löschen", key="dlg_del_grp_submit", type="primary", use_container_width=True):
            delete_group_by_id(group_id)
            st.session_state.active_dialog = None
            st.rerun()

@st.dialog("Voreinstellung umbenennen", on_dismiss=on_dialog_dismiss)
def dialog_rename_preset(preset_id, current_name=None):
    if not current_name:
        current_name = get_preset_name(preset_id)
    input_key = f"dlg_ren_pre_name_{preset_id}"
    new_name = st.text_input("Neuer Name", value=current_name, key=input_key)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_ren_pre_cancel", use_container_width=True):
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        is_disabled = (len(new_name.strip()) == 0)
        if st.button("Speichern", key="dlg_ren_pre_submit", type="primary", disabled=is_disabled, use_container_width=True):
            name_clean = new_name.strip()
            if name_clean:
                p = next((x for x in st.session_state.presets if x["id"] == preset_id), None)
                if p:
                    p["name"] = name_clean
                    write_presets_to_disk()
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            st.rerun()

@st.dialog("Voreinstellung löschen", on_dismiss=on_dialog_dismiss)
def dialog_delete_preset(preset_id, preset_name=None):
    if not preset_name:
        preset_name = get_preset_name(preset_id)
    st.write(f"Möchtest du die Voreinstellung **{preset_name}** wirklich löschen?")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_del_pre_cancel", use_container_width=True):
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.button("Ja, löschen", key="dlg_del_pre_submit", type="primary", use_container_width=True):
            delete_preset_by_id(preset_id)
            st.session_state.active_dialog = None
            st.rerun()

@st.dialog("Voreinstellung kopieren", on_dismiss=on_dialog_dismiss)
def dialog_copy_preset(preset_id, preset_name=None):
    if not preset_name:
        preset_name = get_preset_name(preset_id)
    existing_names = [p["name"] for p in st.session_state.presets]
    default_name = generate_copy_name(preset_name, existing_names)
    input_key = f"dlg_cp_pre_name_{preset_id}"
    new_name = st.text_input("Name der Kopie", value="", placeholder=default_name, key=input_key)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_cp_pre_cancel", use_container_width=True):
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.button("Kopieren", key="dlg_cp_pre_submit", type="primary", use_container_width=True):
            name_clean = new_name.strip() if new_name and new_name.strip() else default_name
            st.session_state.pop(input_key, None)
            st.session_state.active_dialog = None
            copy_preset_by_id(preset_id, name_clean)

def trigger_dialog(dlg_type, item_id=None):
    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
    st.session_state.active_dialog = {"type": dlg_type, "id": item_id}

app_version = load_version()
app_title = f"RPISC Einstellungen {app_version}" if app_version else "RPISC Einstellungen"

st.set_page_config(page_title=app_title, page_icon="🎛️", layout="centered")
preset_data = load_presets()
st.session_state.presets = preset_data["presets"]
st.session_state.groups = preset_data.get("groups", [])
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

if "active_group_id" not in st.session_state or st.session_state.active_group_id is None:
    cur_gid = st.session_state.presets[st.session_state.preset_index].get("groupId")
    st.session_state.active_group_id = cur_gid if cur_gid is not None else GLOBAL_GROUP_ID
elif st.session_state.active_group_id != GLOBAL_GROUP_ID:
    if not any(g["id"] == st.session_state.active_group_id for g in st.session_state.groups):
        st.session_state.active_group_id = GLOBAL_GROUP_ID

update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])

if "static_settings" not in st.session_state:
     st.session_state.static_settings = load_static_settings()
     update_session_state_from_static_settings(st.session_state.static_settings)

if "active_dialog" not in st.session_state:
    st.session_state.active_dialog = None

if st.session_state.get("active_dialog"):
    d = st.session_state.active_dialog
    dtype = d.get("type")
    did = d.get("id")
    if dtype == "create_group":
        dialog_create_group()
    elif dtype == "rename_group":
        dialog_rename_group(did)
    elif dtype == "delete_group":
        dialog_delete_group(did)
    elif dtype == "rename_preset":
        dialog_rename_preset(did)
    elif dtype == "delete_preset":
        dialog_delete_preset(did)
    elif dtype == "copy_preset":
        dialog_copy_preset(did)

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

st.markdown("""
<style>
/* Prevent inner scroll containers */
div[data-testid="stHorizontalBlock"],
div[data-testid="stColumn"] {
    overflow: visible !important;
}

/* Preset rows: Vertically center preset names with adjacent action buttons */
div[data-testid="stHorizontalBlock"]:has([class*="btn_pre_sel_"]) div[data-testid="stMarkdownContainer"],
div[data-testid="stHorizontalBlock"]:has([class*="pop_pre_"]) div[data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 38px !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="btn_pre_sel_"]) div[data-testid="stMarkdownContainer"] p,
div[data-testid="stHorizontalBlock"]:has([class*="pop_pre_"]) div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: normal !important;
}

/* Action buttons: 38px square buttons */
div[class*="st-key-btn_pre_sel_"] button,
div[class*="st-key-mv_up_"] button,
div[class*="st-key-mv_down_"] button,
div[class*="st-key-rm_col_"] button,
button[data-testid="stPopoverButton"] {
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
}
button[data-testid="stPopoverButton"] svg {
    margin: 0 !important;
    padding: 0 !important;
}
button[data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"] {
    display: none !important;
}

/* Group selector row: group selectbox takes all free space, button column takes exactly 38px */
div[data-testid="stHorizontalBlock"]:has([class*="st-key-pop_grp_menu"]) {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-end !important;
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-pop_grp_menu"]) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-pop_grp_menu"]) > div[data-testid="stColumn"]:last-child,
div[data-testid="stColumn"]:has([class*="st-key-pop_grp_menu"]) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 38px !important;
}
[class*="st-key-pop_grp_menu"] {
    width: auto !important;
    min-width: 38px !important;
    display: inline-flex !important;
}

/* Popover menu */
div[data-testid="stPopoverBody"] {
    min-width: 220px !important;
}
div[data-testid="stPopoverBody"] button {
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 8px 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.title(app_title)

active_p = st.session_state.presets[st.session_state.preset_index]
active_gid = active_p.get("groupId")
active_gname = get_group_name(active_gid)
st.info(f"Aktive Voreinstellung: **{active_p['name']}{get_preset_icon_tag(active_p)}** (Gruppe: *{active_gname}*)")

with st.expander("📁 Voreinstellungen & Gruppen", expanded=True):
    group_options = [GLOBAL_GROUP_ID] + [g["id"] for g in st.session_state.groups]
    group_names = {GLOBAL_GROUP_ID: "Global", **{g["id"]: g["name"] for g in st.session_state.groups}}

    # Map each group ID to a unique display label (stable zero-width spaces keyed by sorted ID)
    name_to_gids = {}
    for gid in group_options:
        name = group_names.get(gid, "Global")
        name_to_gids.setdefault(name, []).append(gid)

    group_display_map = {}
    for name, gids in name_to_gids.items():
        if len(gids) == 1:
            group_display_map[gids[0]] = name
        else:
            sorted_gids = sorted(gids)
            for count, gid in enumerate(sorted_gids):
                group_display_map[gid] = name + ("\u200b" * count)

    grp_ver = st.session_state.get("group_selector_ver", 0)
    grp_sel_key = f"group_selector_{grp_ver}"

    def on_group_change():
        selected_gid = st.session_state.get(grp_sel_key)
        if not selected_gid:
            selected_gid = GLOBAL_GROUP_ID
            st.session_state[grp_sel_key] = GLOBAL_GROUP_ID
        st.session_state.active_group_id = selected_gid
        presets_in_new_group = get_presets_in_group(selected_gid)
        cur_preset = st.session_state.presets[st.session_state.preset_index]
        cur_p_gid = cur_preset.get("groupId") or GLOBAL_GROUP_ID
        if cur_p_gid != selected_gid:
            if len(presets_in_new_group) > 0:
                select_preset_by_id(presets_in_new_group[0]["id"])

    grp_idx = 0
    if st.session_state.active_group_id in group_options:
        grp_idx = group_options.index(st.session_state.active_group_id)
    else:
        st.session_state.active_group_id = GLOBAL_GROUP_ID

    if st.session_state.get(grp_sel_key) not in group_options or st.session_state.get(grp_sel_key) != st.session_state.active_group_id:
        st.session_state[grp_sel_key] = st.session_state.active_group_id
        grp_idx = group_options.index(st.session_state.active_group_id)

    col_grp, col_grp_menu = st.columns([8, 1], vertical_alignment="bottom", gap="small", wrap=False)
    with col_grp:
        st.selectbox(
            "Gruppe",
            group_options,
            index=grp_idx,
            format_func=lambda x: group_display_map.get(x, "Global"),
            key=grp_sel_key,
            on_change=on_group_change
        )
    pop_ver = st.session_state.get("popover_ver", 0)
    with col_grp_menu:
        is_custom_group = (st.session_state.active_group_id != GLOBAL_GROUP_ID and st.session_state.active_group_id is not None)
        with st.popover("", key=f"pop_grp_menu_{pop_ver}"):
                st.button("➕ Neue Gruppe", key="btn_grp_new", on_click=trigger_dialog, args=("create_group",), use_container_width=True)
                
                cur_grp_idx = 0
                if is_custom_group:
                    cur_grp_idx = next((i for i, g in enumerate(st.session_state.groups) if g["id"] == st.session_state.active_group_id), 0)

                st.button(
                    "▲ Nach oben",
                    key="btn_grp_up",
                    disabled=(not is_custom_group or cur_grp_idx == 0),
                    on_click=move_group,
                    args=(st.session_state.active_group_id, -1),
                    use_container_width=True
                )
                st.button(
                    "▼ Nach unten",
                    key="btn_grp_down",
                    disabled=(not is_custom_group or cur_grp_idx == len(st.session_state.groups) - 1),
                    on_click=move_group,
                    args=(st.session_state.active_group_id, 1),
                    use_container_width=True
                )
                st.button(
                    "✏️ Gruppe umbenennen",
                    key="btn_grp_rename",
                    disabled=not is_custom_group,
                    on_click=trigger_dialog,
                    args=("rename_group", st.session_state.active_group_id),
                    use_container_width=True
                )
                st.button(
                    "🗑️ Gruppe löschen",
                    key="btn_grp_delete",
                    disabled=not is_custom_group,
                    on_click=trigger_dialog,
                    args=("delete_group", st.session_state.active_group_id),
                    use_container_width=True
                )

    if is_custom_group:
        ungrouped_presets = [p for p in get_presets_in_group(GLOBAL_GROUP_ID) if p["id"] != st.session_state.presets[0]["id"]]
        if len(ungrouped_presets) > 0:
            def on_add_preset_from_global():
                selected_pid = st.session_state.add_from_global_selector
                if selected_pid and selected_pid != NONE_OPTION:
                    p = next((x for x in st.session_state.presets if x["id"] == selected_pid), None)
                    if p:
                        p["groupId"] = st.session_state.active_group_id
                        select_preset_by_id(selected_pid)
                    st.session_state.add_from_global_selector = NONE_OPTION

            options = [NONE_OPTION] + [p["id"] for p in ungrouped_presets]
            format_map = {NONE_OPTION: "Voreinstellung hinzufügen", **{p["id"]: p["name"] for p in ungrouped_presets}}
            st.selectbox(
                "➕ Globale Voreinstellung hinzufügen",
                options,
                index=0,
                format_func=lambda x: format_map.get(x, str(x)),
                key="add_from_global_selector",
                on_change=on_add_preset_from_global
            )

    presets_in_current_group = get_presets_in_group(st.session_state.active_group_id)
    current_grp_title = group_names.get(st.session_state.active_group_id, "Global")

    with st.expander(f"📂 Voreinstellungen in {current_grp_title} ({len(presets_in_current_group)})", expanded=True):
        if len(presets_in_current_group) == 0:
            st.info("Diese Gruppe enthält noch keine Voreinstellungen.")
            if is_custom_group:
                if st.button("➕ Neue Voreinstellung erstellen", key="btn_create_empty_grp_pre"):
                    new_id = str(uuid.uuid4())
                    existing_names = [p["name"] for p in st.session_state.presets]
                    new_name = generate_copy_name("Standard", existing_names)
                    new_preset = {
                        "name": new_name,
                        "id": new_id,
                        "groupId": st.session_state.active_group_id,
                        "values": copy.deepcopy(DYNAMIC_DEFAULTS)
                    }
                    st.session_state.presets.append(new_preset)
                    select_preset_by_id(new_id)
                    st.rerun()
        else:
            for idx_in_grp, p in enumerate(presets_in_current_group):
                is_standard = (p["id"] == st.session_state.presets[0]["id"])
                is_active = (p["id"] == st.session_state.preset_guid)
                icon_tag = get_preset_icon_tag(p)

                col_name, col_actions = st.columns([7, 3], vertical_alignment="center", gap="small", wrap=False)
                with col_name:
                    if is_active:
                        st.markdown(f"**🟢 {p['name']}{icon_tag}**")
                    else:
                        st.markdown(f"⚪ {p['name']}{icon_tag}")
                with col_actions:
                    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
                        if not is_active:
                            st.button(
                                "✅",
                                key=f"btn_pre_sel_{p['id']}",
                                on_click=select_preset_by_id,
                                args=(p["id"],),
                                help="Voreinstellung aktivieren"
                            )
                        with st.popover("", key=f"pop_pre_{p['id']}_{pop_ver}"):
                            can_move_up = False
                            if not is_standard:
                                if st.session_state.active_group_id == GLOBAL_GROUP_ID:
                                    can_move_up = (idx_in_grp > 1)
                                else:
                                    can_move_up = (idx_in_grp > 0)

                            can_move_down = (not is_standard and idx_in_grp < len(presets_in_current_group) - 1)

                            st.button(
                                "▲ Nach oben",
                                key=f"btn_up_{p['id']}",
                                disabled=(not can_move_up),
                                on_click=move_preset_in_group,
                                args=(p["id"], -1),
                                use_container_width=True
                            )
                            st.button(
                                "▼ Nach unten",
                                key=f"btn_down_{p['id']}",
                                disabled=(not can_move_down),
                                on_click=move_preset_in_group,
                                args=(p["id"], 1),
                                use_container_width=True
                            )
                            st.button(
                                "✏️ Umbenennen",
                                key=f"btn_ren_{p['id']}",
                                disabled=is_standard,
                                on_click=trigger_dialog,
                                args=("rename_preset", p["id"]),
                                use_container_width=True
                            )
                            st.button(
                                "📋 Kopieren",
                                key=f"btn_cp_{p['id']}",
                                on_click=trigger_dialog,
                                args=("copy_preset", p["id"]),
                                use_container_width=True
                            )
                            if p.get("groupId") is not None and p.get("groupId") != GLOBAL_GROUP_ID:
                                st.button(
                                    "↩️ In 'Global' verschieben",
                                    key=f"btn_ungrp_{p['id']}",
                                    on_click=unassign_preset,
                                    args=(p["id"],),
                                    use_container_width=True
                                )
                            st.button(
                                "🗑️ Löschen",
                                key=f"btn_del_{p['id']}",
                                disabled=is_standard,
                                on_click=trigger_dialog,
                                args=("delete_preset", p["id"]),
                                use_container_width=True
                            )

    current_guid = st.session_state.preset_guid
    is_master = is_preset_template_master(current_guid)
    cur_p = st.session_state.presets[st.session_state.preset_index]
    cur_p_gid = cur_p.get("groupId") or GLOBAL_GROUP_ID
    if st.session_state.preset_index != 0 and not is_master and len(presets_in_current_group) > 0 and cur_p_gid == st.session_state.active_group_id:
        std_id = st.session_state.presets[0]["id"]
        filtered_values = [
            v for v in st.session_state.presets
            if v["id"] != st.session_state.preset_guid
            and v["id"] != std_id
            and v["values"].get("templateId") is None
        ]
        options = [NONE_OPTION] + [v["id"] for v in filtered_values]
        if "templateId" not in st.session_state:
            st.session_state.templateId = None

        cur_tmpl_key = f"template_select_{st.session_state.preset_guid}"
        template_guid = cur_p["values"].get("templateId")

        default_index = 0
        if template_guid is not None and template_guid in options:
            default_index = options.index(template_guid)

        def on_template_changed():
            selected_guid = st.session_state.get(cur_tmpl_key)
            if selected_guid == NONE_OPTION:
                st.session_state.templateId = None
            else:
                st.session_state.templateId = selected_guid
        
            save_presets()

        st.selectbox(
            "Vorlage für aktive Voreinstellung",
            options,
            index=default_index,
            format_func=lambda x: "Deaktiviert" if x is NONE_OPTION else get_preset_name(x),
            key=cur_tmpl_key,
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

if len(presets_in_current_group) > 0:
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
                    on_change=save_presets
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
                on_change=save_presets
            )

with st.expander("Globale Einstellungen", expanded=False):
    st.number_input("FPS", min_value=1, max_value=400, step=1, format="%d", key="fps", on_change=save_static_config)
    st.number_input("LED Anzahl", min_value=2, max_value=99999, step=1, format="%d", key="ledCount", on_change=save_static_config)
