import streamlit as st
import json
import os
import copy
import uuid

STATIC_CONFIG_FILE = "static_settings.json"
PRESETS_FILE = "presets.json"

AUDIO_MODES = {
    0: "Dynamisch",
    1: "Simple"
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
    "resetEffectAfterTransition": False
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
    "ledCount": 300
}

NONE_OPTION = "__none__"

def load_presets():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r") as f:
                data = json.load(f)
                if "presets" not in data or len(data["presets"]) == 0:
                    return create_default_presets_file()
                return data
        except:
            return create_default_presets_file()
    else:
        return create_default_presets_file()

def load_static_settings():
    if os.path.exists(STATIC_CONFIG_FILE):
        try:
            with open(STATIC_CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return create_default_static_settings_file()
    else:
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
    with open(PRESETS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def write_static_config_to_disk(values):
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

def get_current_preset_values_from_state():
    sync_widgets_to_colors_list()
   
    aud_mode_id = next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode) 
    eff_mode_id = next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
    col_mode_id = next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)

    return {
        "colors": st.session_state.colors,
        "useRainbow": st.session_state.useRainbow,
        "effectOrigin": st.session_state.effectOrigin,
        "speed": st.session_state.speed,
        "minFreq": st.session_state.minFreq,
        "maxFreq": st.session_state.maxFreq,
        "fade": st.session_state.fade,
        "fadeOverTime": st.session_state.fadeOverTime,
        "staticSpectrum": st.session_state.staticSpectrum,
        "spectrumSections": st.session_state.spectrumSections,
        "bouncyWave": st.session_state.bouncyWave,
        "saturate": st.session_state.saturate,
        "saturateThreshold": st.session_state.saturateThreshold,
        "meanValueBufferSize": st.session_state.meanValueBufferSize,
        "meanValueThreshold": st.session_state.meanValueThreshold,
        "audioMode": aud_mode_id,
        "effectMode": eff_mode_id,
        "colorMode": col_mode_id,
        "minFreqAmplitude": st.session_state.minFreqAmplitude,
        "maxFreqAmplitude": st.session_state.maxFreqAmplitude,
        "colorIncreaseFactor": st.session_state.colorIncreaseFactor,
        "valueIncreaseFactor": st.session_state.valueIncreaseFactor,
        "colorTransition": st.session_state.colorTransition,
        "valueColorBias": st.session_state.valueColorBias,
        "colorWaveOrigin": st.session_state.colorWaveOrigin,
        "colorWaveSpeed": st.session_state.colorWaveSpeed,
        "colorWaveSize": st.session_state.colorWaveSize,
        "colorWaveInwards": st.session_state.colorWaveInwards,
        "noiseAmount": st.session_state.noiseAmount,
        "noiseSmoothing": st.session_state.noiseSmoothing,
        "getAlphaFromValue": st.session_state.getAlphaFromValue,
        "colorOverflow": st.session_state.colorOverflow,
        "brightness": st.session_state.brightness,
        "gamma": st.session_state.gamma,
        "effectRepeats": st.session_state.effectRepeats,
        "acceleration": st.session_state.acceleration,
        "particleSize": st.session_state.particleSize,
        "patternSplits": st.session_state.patternSplits,
        "patternSpread": st.session_state.patternSpread,
        "patternFlip": st.session_state.patternFlip,
        "patternCenter": st.session_state.patternCenter,
        "patternSectionSizeMod": st.session_state.patternSectionSizeMod,
        "fftSize": st.session_state.fftSize,
        "bpmLimit": st.session_state.bpmLimit,
        "audioResponseCurve": st.session_state.audioResponseCurve,
        "audioPeakHoldTimeMs": st.session_state.audioPeakHoldTimeMs,
        "nextEffectId": st.session_state.nextEffectId,
        "effectDurationMs": st.session_state.effectDurationMs,
        "effectTransitionDurationMs": st.session_state.effectTransitionDurationMs,
        "effectTransitionWarmupDuration": st.session_state.effectTransitionWarmupDuration,
        "resetEffectAfterTransition": st.session_state.resetEffectAfterTransition
    }

def update_session_state_from_preset(preset_data):
    st.session_state.preset_guid = preset_data["id"]
    vals = preset_data["values"]
    st.session_state.colors = copy.deepcopy(vals.get("colors", DYNAMIC_DEFAULTS["colors"]))
    for i, c in enumerate(st.session_state.colors):
        st.session_state[f"color_{i}"] = c["color"]
        st.session_state[f"threshold_{i}"] = c["threshold"]

    st.session_state.useRainbow = vals.get("useRainbow", DYNAMIC_DEFAULTS["useRainbow"])
    st.session_state.effectOrigin = vals.get("effectOrigin", DYNAMIC_DEFAULTS["effectOrigin"])
    st.session_state.speed = vals.get("speed", DYNAMIC_DEFAULTS["speed"])
    st.session_state.minFreq = vals.get("minFreq", DYNAMIC_DEFAULTS["minFreq"])
    st.session_state.maxFreq = vals.get("maxFreq", DYNAMIC_DEFAULTS["maxFreq"])
    st.session_state.fade = vals.get("fade", DYNAMIC_DEFAULTS["fade"])
    st.session_state.fadeOverTime = vals.get("fadeOverTime", DYNAMIC_DEFAULTS["fadeOverTime"])
    st.session_state.bouncyWave = vals.get("bouncyWave", DYNAMIC_DEFAULTS["bouncyWave"])
    st.session_state.staticSpectrum = vals.get("staticSpectrum", DYNAMIC_DEFAULTS["staticSpectrum"])
    st.session_state.spectrumSections = vals.get("spectrumSections", DYNAMIC_DEFAULTS["spectrumSections"])
    st.session_state.saturate = vals.get("saturate", DYNAMIC_DEFAULTS["saturate"])
    st.session_state.saturateThreshold = vals.get("saturateThreshold", DYNAMIC_DEFAULTS["saturateThreshold"])
    st.session_state.meanValueBufferSize = vals.get("meanValueBufferSize", DYNAMIC_DEFAULTS["meanValueBufferSize"])
    st.session_state.meanValueThreshold = vals.get("meanValueThreshold", DYNAMIC_DEFAULTS["meanValueThreshold"])
    st.session_state.minFreqAmplitude = vals.get("minFreqAmplitude", DYNAMIC_DEFAULTS["minFreqAmplitude"])
    st.session_state.maxFreqAmplitude = vals.get("maxFreqAmplitude", DYNAMIC_DEFAULTS["maxFreqAmplitude"])
    st.session_state.colorIncreaseFactor = vals.get("colorIncreaseFactor", DYNAMIC_DEFAULTS["colorIncreaseFactor"])
    st.session_state.valueIncreaseFactor = vals.get("valueIncreaseFactor", DYNAMIC_DEFAULTS["valueIncreaseFactor"])
    st.session_state.colorTransition = vals.get("colorTransition", DYNAMIC_DEFAULTS["colorTransition"])
    st.session_state.valueColorBias = vals.get("valueColorBias", DYNAMIC_DEFAULTS["valueColorBias"])
    st.session_state.getAlphaFromValue = vals.get("getAlphaFromValue", DYNAMIC_DEFAULTS["getAlphaFromValue"])
    st.session_state.colorOverflow = vals.get("colorOverflow", DYNAMIC_DEFAULTS["colorOverflow"])
    st.session_state.colorWaveOrigin = vals.get("colorWaveOrigin", DYNAMIC_DEFAULTS["colorWaveOrigin"])
    st.session_state.colorWaveSpeed = vals.get("colorWaveSpeed", DYNAMIC_DEFAULTS["colorWaveSpeed"])
    st.session_state.colorWaveSize = vals.get("colorWaveSize", DYNAMIC_DEFAULTS["colorWaveSize"])
    st.session_state.colorWaveInwards = vals.get("colorWaveInwards", DYNAMIC_DEFAULTS["colorWaveInwards"])
    st.session_state.noiseAmount = vals.get("noiseAmount", DYNAMIC_DEFAULTS["noiseAmount"])
    st.session_state.noiseSmoothing = vals.get("noiseSmoothing", DYNAMIC_DEFAULTS["noiseSmoothing"])
    st.session_state.brightness = vals.get("brightness", DYNAMIC_DEFAULTS["brightness"])
    st.session_state.gamma = vals.get("gamma", DYNAMIC_DEFAULTS["gamma"])
    st.session_state.effectRepeats = vals.get("effectRepeats", DYNAMIC_DEFAULTS["effectRepeats"])
    st.session_state.acceleration = vals.get("acceleration", DYNAMIC_DEFAULTS["acceleration"])
    st.session_state.particleSize = vals.get("particleSize", DYNAMIC_DEFAULTS["particleSize"])
    st.session_state.patternSplits = vals.get("patternSplits", DYNAMIC_DEFAULTS["patternSplits"])
    st.session_state.patternSpread = vals.get("patternSpread", DYNAMIC_DEFAULTS["patternSpread"])
    st.session_state.patternFlip = vals.get("patternFlip", DYNAMIC_DEFAULTS["patternFlip"])
    st.session_state.patternCenter = vals.get("patternCenter", DYNAMIC_DEFAULTS["patternCenter"])
    st.session_state.patternSectionSizeMod = vals.get("patternSectionSizeMod", DYNAMIC_DEFAULTS["patternSectionSizeMod"])
    st.session_state.fftSize = vals.get("fftSize", DYNAMIC_DEFAULTS["fftSize"])
    st.session_state.bpmLimit = vals.get("bpmLimit", DYNAMIC_DEFAULTS["bpmLimit"])
    st.session_state.audioResponseCurve = vals.get("audioResponseCurve", DYNAMIC_DEFAULTS["audioResponseCurve"])
    st.session_state.audioPeakHoldTimeMs = vals.get("audioPeakHoldTimeMs", DYNAMIC_DEFAULTS["audioPeakHoldTimeMs"])
    st.session_state.nextEffectId = vals.get("nextEffectId", DYNAMIC_DEFAULTS["nextEffectId"])
    st.session_state.effectDurationMs = vals.get("effectDurationMs", DYNAMIC_DEFAULTS["effectDurationMs"])
    st.session_state.effectTransitionDurationMs = vals.get("effectTransitionDurationMs", DYNAMIC_DEFAULTS["effectTransitionDurationMs"])
    st.session_state.effectTransitionWarmupDuration = vals.get("effectTransitionWarmupDuration", DYNAMIC_DEFAULTS["effectTransitionWarmupDuration"])
    st.session_state.resetEffectAfterTransition = vals.get("resetEffectAfterTransition", DYNAMIC_DEFAULTS["resetEffectAfterTransition"])

    a_mode = vals.get("audioMode", DYNAMIC_DEFAULTS["audioMode"])
    e_mode = vals.get("effectMode", DYNAMIC_DEFAULTS["effectMode"])
    c_mode = vals.get("colorMode", DYNAMIC_DEFAULTS["colorMode"])
    st.session_state.audioMode = AUDIO_MODES.get(a_mode, AUDIO_MODES[0])
    st.session_state.effectMode = EFFECT_MODES.get(e_mode, EFFECT_MODES[0])
    st.session_state.colorMode = COLOR_MODES.get(c_mode, COLOR_MODES[0])
    st.session_state.rename_input = preset_data["name"]

def update_session_state_from_static_settings(static_settings_data):
    st.session_state.fps = static_settings_data.get("fps", STATIC_DEFAULTS["fps"])
    st.session_state.ledCount = static_settings_data.get("ledCount", STATIC_DEFAULTS["ledCount"])

def save_presets():
    current_values = get_current_preset_values_from_state()
    
    idx = st.session_state.preset_index
    st.session_state.presets[idx]["values"] = current_values
    
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
            if st.session_state.presets[i]["values"]["nextEffectId"] == removed_preset["id"]:
                st.session_state.presets[i]["values"]["nextEffectId"] = None                

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

def cb_remove_color():
    if len(st.session_state.colors) > 2:
        sync_widgets_to_colors_list()
        st.session_state.colors.pop()

        old_idx = len(st.session_state.colors) 
        if f"color_{old_idx}" in st.session_state: del st.session_state[f"color_{old_idx}"]
        if f"threshold_{old_idx}" in st.session_state: del st.session_state[f"threshold_{old_idx}"]
        
        save_presets()

st.set_page_config(page_title="RPISC Einstellungen", page_icon="🎛️", layout="centered")

if "presets" not in st.session_state:
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

st.markdown("""
<style>
@media (max-width: 768px) {
    div[data-testid="stSlider"] {
        width: 80% !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("RPISC Einstellungen")
st.info(f"Aktive Voreinstellung: **{st.session_state.presets[st.session_state.preset_index]['name']}**")

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

curr_preset = st.session_state.presets[st.session_state.preset_index]["values"]

def get_audio_mode_id():
    return next(k for k, v in AUDIO_MODES.items() if v == st.session_state.audioMode)
def get_effect_mode_id():
    return next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
def get_color_mode_id():
    return next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)

with st.expander("Audio", expanded=False):
    st.selectbox("Audiomodus", list(AUDIO_MODES.values()), key="audioMode", on_change=save_presets)
    st.slider("Min. Lautstärke", 0.0, 50.0, step=0.01, key="minFreqAmplitude", on_change=save_presets)
    st.slider("Max. Lautstärke", 0.0, 50.0, step=0.01, key="maxFreqAmplitude", on_change=save_presets)

    st.session_state.meanValueBufferSize = curr_preset.get("meanValueBufferSize")
    st.session_state.meanValueThreshold = curr_preset.get("meanValueThreshold")

    if get_audio_mode_id() == 0:
        st.slider("Vergleichswert Puffergröße", 1, 100, step=1, key="meanValueBufferSize", on_change=save_presets)
        st.slider("Vergleichswert Grenzwert", 0.01, 1.0, step=0.01, key="meanValueThreshold", on_change=save_presets)

    st.slider("BPM Limit", -1, 999, step=1, key="bpmLimit", on_change=save_presets)
    st.slider("Audiowert Skalierung", 0.01, 5.00, step=0.01, key="audioResponseCurve", on_change=save_presets)
    st.slider("Min. Peakdauer in ms", 0, 999, step=1, key="audioPeakHoldTimeMs", on_change=save_presets)
    st.number_input("Min. Frequenz in Hz", min_value=0, max_value=20000, step=1, format="%d", key="minFreq", on_change=save_presets)
    st.number_input("Max. Frequenz in Hz", min_value=0, max_value=20000, step=1, format="%d", key="maxFreq", on_change=save_presets)
    st.number_input("Audio Puffer Größe", min_value=2, max_value=50000, step=1, format="%d", key="fftSize", on_change=save_presets)

with st.expander("Effekt", expanded=False):
    st.selectbox("Effektmodus", list(EFFECT_MODES.values()), key="effectMode", on_change=save_presets)
    st.number_input("Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="effectOrigin", on_change=save_presets)
    st.slider("Effekt Beschleunigung", 0.0, 5.0, step=0.01, key="acceleration", on_change=save_presets)

    st.session_state.speed = curr_preset.get("speed")
    if get_effect_mode_id() in (2, 4):
        st.slider("Effekt Geschwindigkeit", 1, 600, key="speed", on_change=save_presets)

    st.slider("Effekt Verstärkung", 0.1, 10.0, key="valueIncreaseFactor", on_change=save_presets)

    st.session_state.staticSpectrum = curr_preset.get("staticSpectrum")
    st.session_state.spectrumSections = curr_preset.get("spectrumSections")
    if get_effect_mode_id() == 3:
        st.toggle("Statisches Spektrum", key="staticSpectrum", on_change=save_presets)
        st.slider("Spektrum Sektionen", 1, 50, step=1, key="spectrumSections", on_change=save_presets)

    st.session_state.particleSize = curr_preset.get("particleSize")
    if get_effect_mode_id() == 4:
        st.slider("Partikel Größe", 1, 1000, step=1, key="particleSize", on_change=save_presets)

    st.session_state.bouncyWave = curr_preset.get("bouncyWave")
    if get_effect_mode_id() == 2:
        st.toggle("Abprallen", key="bouncyWave", on_change=save_presets)

    st.session_state.fadeOverTime = curr_preset.get("fadeOverTime")
    if get_effect_mode_id() == 2 or get_effect_mode_id() == 8:
        st.slider("Verblassung nach Zeit", -0.999, 0.999, step=0.001, key="fadeOverTime", on_change=save_presets)

    st.slider("Verblassung", 0.001, 0.999, step=0.001, key="fade", on_change=save_presets)
    st.slider("Sättigung", 0.01, 1.0, step=0.01, key="saturate", on_change=save_presets)
    st.slider("Sättigungs Grenzwert", 0.0, 1.0, step=0.01, key="saturateThreshold", on_change=save_presets)

with st.expander("Muster", expanded=False):
    st.slider("Effekt Wiederholungen", 1, 50, step=1, key="effectRepeats", on_change=save_presets)
    st.slider("Muster Teilungen", 0, 50, step=1, key="patternSplits", on_change=save_presets)
    st.slider("Jedes N'te Muster verdrehen", -1, 50, step=1, key="patternFlip", on_change=save_presets)
    st.number_input("Muster Verteilung", min_value=0, max_value=999999, step=1, format="%d", key="patternSpread", on_change=save_presets)
    st.number_input("Muster Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="patternCenter", on_change=save_presets)
    st.slider("Muster Größenveränderung", -5.00, 5.00, step=0.01, key="patternSectionSizeMod", on_change=save_presets)   

with st.expander("Sequenzierung", expanded=False):
    guid_to_obj = {v["id"]: v for v in st.session_state.presets}
    current_guid = st.session_state.preset_guid
    filtered_values = [v for v in st.session_state.presets if v["id"] != current_guid]
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

    st.number_input("Effekt Dauer in ms", min_value=1, max_value=99999999, step=1, format="%d", key="effectDurationMs", on_change=save_presets)
    st.number_input("Effekt Übergangsdauer in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionDurationMs", on_change=save_presets)
    st.number_input("Übergangs Vorlaufzeit in ms", min_value=0, max_value=99999999, step=1, format="%d", key="effectTransitionWarmupDuration", on_change=save_presets)
    st.toggle("Effekt nach Übergang zurücksetzen", key="resetEffectAfterTransition", on_change=save_presets)

with st.expander("Farben", expanded=False):
    st.selectbox("Farbmodus", list(COLOR_MODES.values()), key="colorMode", on_change=save_presets)
    st.slider("Helligkeit", 0.0, 5.0, step=0.01, key="brightness", on_change=save_presets)
    st.slider("Gamma", 0.0, 5.0, step=0.01, key="gamma", on_change=save_presets)
    st.slider("Farbverstärkung", 0.1, 20.0, key="colorIncreaseFactor", on_change=save_presets)
    st.slider("Farbverlauf", 0.00, 0.50, key="colorTransition", on_change=save_presets)

    st.session_state.valueColorBias = curr_preset.get("valueColorBias")
    st.session_state.getAlphaFromValue = curr_preset.get("getAlphaFromValue")
    if get_color_mode_id() != 0:
        st.slider("Wert / Farb Verschmierung", 0.00, 1.00, step=0.01, key="valueColorBias", on_change=save_presets)
        st.toggle("Transparenz aus Wert", key="getAlphaFromValue", on_change=save_presets)

    st.session_state.colorWaveOrigin = curr_preset.get("colorWaveOrigin")
    st.session_state.colorWaveSpeed = curr_preset.get("colorWaveSpeed")
    st.session_state.colorWaveSize = curr_preset.get("colorWaveSize")
    st.session_state.colorWaveInwards = curr_preset.get("colorWaveInwards")

    if get_color_mode_id() == 4:
        st.number_input("Farbwellen Zentrum", min_value=0, max_value=999999, step=1, format="%d", key="colorWaveOrigin", on_change=save_presets)
        st.number_input("Farbwellen Größe", min_value=1, max_value=999999, step=1, format="%d", key="colorWaveSize", on_change=save_presets)
        st.slider("Farbwellen Geschwindigkeit", 1, 600, key="colorWaveSpeed", on_change=save_presets)
        st.toggle("Farbwellen Richtung (nach Außen / Innen)", key="colorWaveInwards", on_change=save_presets)
    
    st.slider("Zufallsfarben Faktor", 0.00, 1.00, step=0.01, key="noiseAmount", on_change=save_presets)
    st.slider("Zufallsfarben Glättung", 0.00, 1.00, step=0.01, key="noiseSmoothing", on_change=save_presets)

    st.toggle("Farbüberfluss", key="colorOverflow", on_change=save_presets)
    st.toggle("Regenbogen Farben", key="useRainbow", on_change=save_presets)

    col_add, col_remove = st.columns(2)

    with col_add:
        st.button("➕ Farbe hinzufügen", on_click=cb_add_color)

    with col_remove:
        st.button("➖ Letzte Farbe entfernen", disabled=(len(st.session_state.colors) <= 2), on_click=cb_remove_color)

    for i, c in enumerate(st.session_state.colors):
        label = "Hintergrund Farbe" if i == 0 else f"Farbe {i}"
    
        if f"color_{i}" not in st.session_state:
            st.session_state[f"color_{i}"] = c["color"]
        if f"threshold_{i}" not in st.session_state:
            st.session_state[f"threshold_{i}"] = c["threshold"]
        
        col_c, col_t = st.columns([1, 2])
        with col_c:
            st.color_picker(
                label,
                key=f"color_{i}",
                on_change=save_presets
            )
        with col_t:
            st.slider(
                "Grenzwert",
                min_value=0.00,
                max_value=1.00,
                step=0.01,
                key=f"threshold_{i}",
                label_visibility="visible",
                on_change=save_presets
            )

with st.expander("Globale Einstellungen", expanded=False):
    st.number_input("FPS", min_value=1, max_value=400, step=1, format="%d", key="fps", on_change=save_static_config)
    st.number_input("LED Anzahl", min_value=2, max_value=99999, step=1, format="%d", key="ledCount", on_change=save_static_config)
