import streamlit as st
import json
import os
import copy

CONFIG_FILE = "audio_params.json"
PRESETS_FILE = "presets.json"

EFFECT_MODES = {
    0: "Pulsate",
    1: "Line",
    2: "Wave",
    3: "Spectrum",
    4: "Particel",
    5: "Static",
    6: "External"
}

COLOR_MODES = {
    0: "By value",
    1: "By index",
    2: "By distance to center",
    3: "By distance to border",
    4: "By color wave",
    5: "By value islands"
}

DEFAULTS = {
    "colors": [
        { "color": "#000000", "threshold": 0.25 },
        { "color": "#FFFFFF", "threshold": 1.0 }
    ],
    "useRainbow": False,
    "effectOrigin": 139,
    "speed": 300,
    "minFreq": 0,
    "maxFreq": 180,
    "fade": 0.6,
    "saturate": 0.6,
    "saturateThreshold": 0.3,
    "meanValueBufferSize": 20,
    "meanValueThreshold": 0.3,
    "effectMode": 0,
    "colorMode": 0,
    "minFreqAmplitude": 0.1,
    "colorIncreaseFactor": 1.0,
    "valueIncreaseFactor": 1.0,
    "colorTransition": 0.25,
    "valueColorBias": 0.0,
    "getAlphaFromValue": False,
    "colorOverflow": False,
    "colorWaveOrigin": 139,
    "colorWaveSpeed": 50,
    "colorWaveSize": 100,
    "colorWaveInwards": False,
    "noiseAmount": 0.00,
    "noiseSmoothing": 1.00,
    "brightness": 1.00,
    "gamma": 1.00
}

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

def create_default_presets_file():
    initial_data = {
        "selected_index": 0,
        "presets": [
            {
                "name": "Default",
                "values": copy.deepcopy(DEFAULTS)
            }
        ]
    }
    with open(PRESETS_FILE, "w") as f:
        json.dump(initial_data, f, indent=2)
    return initial_data

def save_presets_to_disk():
    data = {
        "selected_index": st.session_state.preset_index,
        "presets": st.session_state.presets
    }
    with open(PRESETS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def save_config(values):
    with open(CONFIG_FILE, "w") as f:
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

def get_current_values_from_state():
    sync_widgets_to_colors_list()
    
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
        "saturate": st.session_state.saturate,
        "saturateThreshold": st.session_state.saturateThreshold,
        "meanValueBufferSize": st.session_state.meanValueBufferSize,
        "meanValueThreshold": st.session_state.meanValueThreshold,
        "effectMode": eff_mode_id,
        "colorMode": col_mode_id,
        "minFreqAmplitude": st.session_state.minFreqAmplitude,
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
        "gamma": st.session_state.gamma
    }

def update_session_state_from_preset(preset_data):
    vals = preset_data["values"]
    st.session_state.colors = copy.deepcopy(vals.get("colors", DEFAULTS["colors"]))
    for i, c in enumerate(st.session_state.colors):
        st.session_state[f"color_{i}"] = c["color"]
        st.session_state[f"threshold_{i}"] = c["threshold"]

    st.session_state.useRainbow = vals.get("useRainbow", DEFAULTS["useRainbow"])
    st.session_state.effectOrigin = vals.get("effectOrigin", DEFAULTS["effectOrigin"])
    st.session_state.speed = vals.get("speed", DEFAULTS["speed"])
    st.session_state.minFreq = vals.get("minFreq", DEFAULTS["minFreq"])
    st.session_state.maxFreq = vals.get("maxFreq", DEFAULTS["maxFreq"])
    st.session_state.fade = vals.get("fade", DEFAULTS["fade"])
    st.session_state.saturate = vals.get("saturate", DEFAULTS["saturate"])
    st.session_state.saturateThreshold = vals.get("saturateThreshold", DEFAULTS["saturateThreshold"])
    st.session_state.meanValueBufferSize = vals.get("meanValueBufferSize", DEFAULTS["meanValueBufferSize"])
    st.session_state.meanValueThreshold = vals.get("meanValueThreshold", DEFAULTS["meanValueThreshold"])
    st.session_state.minFreqAmplitude = vals.get("minFreqAmplitude", DEFAULTS["minFreqAmplitude"])
    st.session_state.colorIncreaseFactor = vals.get("colorIncreaseFactor", DEFAULTS["colorIncreaseFactor"])
    st.session_state.valueIncreaseFactor = vals.get("valueIncreaseFactor", DEFAULTS["valueIncreaseFactor"])
    st.session_state.colorTransition = vals.get("colorTransition", DEFAULTS["colorTransition"])
    st.session_state.valueColorBias = vals.get("valueColorBias", DEFAULTS["valueColorBias"])
    st.session_state.getAlphaFromValue = vals.get("getAlphaFromValue", DEFAULTS["getAlphaFromValue"])
    st.session_state.colorOverflow = vals.get("colorOverflow", DEFAULTS["colorOverflow"])
    st.session_state.colorWaveOrigin = vals.get("colorWaveOrigin", DEFAULTS["colorWaveOrigin"])
    st.session_state.colorWaveSpeed = vals.get("colorWaveSpeed", DEFAULTS["colorWaveSpeed"])
    st.session_state.colorWaveSize = vals.get("colorWaveSize", DEFAULTS["colorWaveSize"])
    st.session_state.colorWaveInwards = vals.get("colorWaveInwards", DEFAULTS["colorWaveInwards"])
    st.session_state.noiseAmount = vals.get("noiseAmount", DEFAULTS["noiseAmount"])
    st.session_state.noiseSmoothing = vals.get("noiseSmoothing", DEFAULTS["noiseSmoothing"])
    st.session_state.brightness = vals.get("brightness", DEFAULTS["brightness"])
    st.session_state.gamma = vals.get("gamma", DEFAULTS["gamma"])

    e_mode = vals.get("effectMode", DEFAULTS["effectMode"])
    c_mode = vals.get("colorMode", DEFAULTS["colorMode"])
    st.session_state.effectMode = EFFECT_MODES.get(e_mode, EFFECT_MODES[0])
    st.session_state.colorMode = COLOR_MODES.get(c_mode, COLOR_MODES[0])
    st.session_state.rename_input = preset_data["name"]

def save_params():
    current_values = get_current_values_from_state()
    
    idx = st.session_state.preset_index
    st.session_state.presets[idx]["values"] = current_values
    
    save_presets_to_disk()
    save_config(current_values)

def cb_create_preset():
    save_params()
    
    current_data = copy.deepcopy(st.session_state.presets[st.session_state.preset_index])
    current_data["name"] = f"{current_data['name']} (Copy)"
    st.session_state.presets.append(current_data)
 
    st.session_state.preset_index = len(st.session_state.presets) - 1
    update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
    
    save_presets_to_disk()
    save_config(st.session_state.presets[st.session_state.preset_index]["values"])

def cb_delete_preset():
    if len(st.session_state.presets) > 1 and st.session_state.preset_index != 0:
        st.session_state.presets.pop(st.session_state.preset_index)
        st.session_state.preset_index = max(0, st.session_state.preset_index - 1)
        update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
        save_presets_to_disk()
        save_config(st.session_state.presets[st.session_state.preset_index]["values"])

def cb_add_color():
    sync_widgets_to_colors_list()
    st.session_state.colors.append({ "color": "#ffffff", "threshold": 1.0 })
    
    new_idx = len(st.session_state.colors) - 1
    st.session_state[f"color_{new_idx}"] = "#ffffff"
    st.session_state[f"threshold_{new_idx}"] = 1.0
    
    save_params()

def cb_remove_color():
    if len(st.session_state.colors) > 2:
        sync_widgets_to_colors_list()
        st.session_state.colors.pop()

        old_idx = len(st.session_state.colors) 
        if f"color_{old_idx}" in st.session_state: del st.session_state[f"color_{old_idx}"]
        if f"threshold_{old_idx}" in st.session_state: del st.session_state[f"threshold_{old_idx}"]
        
        save_params()

st.set_page_config(page_title="RPISC Settings", page_icon="🎛️", layout="centered")

if "presets" not in st.session_state:
    preset_data = load_presets()
    st.session_state.presets = preset_data["presets"]
    st.session_state.preset_index = preset_data.get("selected_index", 0)
    
    if st.session_state.preset_index >= len(st.session_state.presets):
        st.session_state.preset_index = 0
    
    if "rename_input" not in st.session_state:
        st.session_state.rename_input = st.session_state.presets[st.session_state.preset_index]["name"]

    update_session_state_from_preset(st.session_state.presets[st.session_state.preset_index])
    save_config(st.session_state.presets[st.session_state.preset_index]["values"])

st.title("RPISC Settings")
st.info(f"Active preset: **{st.session_state.presets[st.session_state.preset_index]['name']}**")

with st.expander("📂 Preset management", expanded=True):
    
    preset_names = [p["name"] for p in st.session_state.presets]
    
    def on_preset_change():
        new_index = st.session_state.preset_selector_idx
        st.session_state.preset_index = new_index
        update_session_state_from_preset(st.session_state.presets[new_index])
        save_presets_to_disk()
        save_config(st.session_state.presets[new_index]["values"])

    st.selectbox(
        "Select preset",
        range(len(preset_names)),
        format_func=lambda x: preset_names[x],
        index=st.session_state.preset_index,
        key="preset_selector_idx",
        on_change=on_preset_change
    )

    col_name, col_actions = st.columns([2, 1])

    with col_name:
        def on_rename():
            new_name = st.session_state.rename_input
            if new_name:
                st.session_state.presets[st.session_state.preset_index]["name"] = new_name
                save_presets_to_disk()

        st.text_input(
            "Rename preset",
            key="rename_input",
            disabled=(st.session_state.preset_index == 0),
            on_change=on_rename
        )

    with col_actions:
        st.write("") 
        st.write("") 
        
        st.button("➕ New (Copy)", on_click=cb_create_preset)
        st.button("🗑️ Delete", disabled=(st.session_state.preset_index == 0), on_click=cb_delete_preset)

st.divider()
curr_preset = st.session_state.presets[st.session_state.preset_index]["values"]

def get_effect_mode_id():
    return next(k for k, v in EFFECT_MODES.items() if v == st.session_state.effectMode)
def get_color_mode_id():
    return next(k for k, v in COLOR_MODES.items() if v == st.session_state.colorMode)

st.selectbox("Effect mode", list(EFFECT_MODES.values()), key="effectMode", on_change=save_params)
st.selectbox("Color mode", list(COLOR_MODES.values()), key="colorMode", on_change=save_params)

st.session_state.effectOrigin = curr_preset.get("effectOrigin")
if get_effect_mode_id() in (1, 2, 3) or get_color_mode_id() == 2:
    st.slider("Center", 0, 277, key="effectOrigin", on_change=save_params)    

st.session_state.speed = curr_preset.get("speed")
if get_effect_mode_id() in (2, 4):
    st.slider("Effect speed", 1, 600, key="speed", on_change=save_params)

st.slider("Min. frequency", 0, 12000, key="minFreq", on_change=save_params)
st.slider("Max. frequency", 0, 12000, key="maxFreq", on_change=save_params)
st.slider("Color fade", 0.001, 0.999, step=0.001, key="fade", on_change=save_params)
st.slider("Color saturate", 0.01, 1.0, step=0.01, key="saturate", on_change=save_params)
st.slider("Saturate threshold", 0.0, 1.0, step=0.01, key="saturateThreshold", on_change=save_params)
st.slider("Mean value buffer size", 1, 100, step=1, key="meanValueBufferSize", on_change=save_params)
st.slider("Mean value threshold", 0.01, 1.0, step=0.01, key="meanValueThreshold", on_change=save_params)
st.slider("Min. volume", 0.0, 50.0, key="minFreqAmplitude", on_change=save_params)
st.slider("Intensity increase", 0.1, 10.0, key="valueIncreaseFactor", on_change=save_params)

st.divider()
st.subheader("Colors")

st.slider("Brightness", 0.0, 1.0, step=0.01, key="brightness", on_change=save_params)
st.slider("Gamma", 0.0, 5.0, step=0.01, key="gamma", on_change=save_params)
st.slider("Color increase", 0.1, 20.0, key="colorIncreaseFactor", on_change=save_params)
st.slider("Color transition", 0.00, 0.50, key="colorTransition", on_change=save_params)

st.session_state.valueColorBias = curr_preset.get("valueColorBias")
st.session_state.getAlphaFromValue = curr_preset.get("getAlphaFromValue")
if get_color_mode_id() != 0:
    st.slider("Value / Color bias", 0.00, 1.00, step=0.01, key="valueColorBias", on_change=save_params)
    st.toggle("Use alpha from value", key="getAlphaFromValue", on_change=save_params)

st.session_state.colorWaveOrigin = curr_preset.get("colorWaveOrigin")
st.session_state.colorWaveSpeed = curr_preset.get("colorWaveSpeed")
st.session_state.colorWaveSize = curr_preset.get("colorWaveSize")
st.session_state.colorWaveInwards = curr_preset.get("colorWaveInwards")
if get_color_mode_id() == 4:
    st.slider("Color wave center", 0, 277, key="colorWaveOrigin", on_change=save_params)
    st.slider("Color wave speed", 1, 600, key="colorWaveSpeed", on_change=save_params)
    st.slider("Color wave size", 1, 1000, step=1, key="colorWaveSize", on_change=save_params)
    st.toggle("Color wave direction (outwards / inwards)", key="colorWaveInwards", on_change=save_params)
    
st.slider("Color noise", 0.00, 1.00, step=0.01, key="noiseAmount", on_change=save_params)
st.slider("Color noise smoothing", 0.00, 1.00, step=0.01, key="noiseSmoothing", on_change=save_params)

st.toggle("Color overflow", key="colorOverflow", on_change=save_params)
st.toggle("Use rainbow colors", key="useRainbow", on_change=save_params)

col_add, col_remove = st.columns(2)

with col_add:
    st.button("➕ Add color", on_click=cb_add_color)

with col_remove:
    st.button("➖ Remove last color", disabled=(len(st.session_state.colors) <= 2), on_click=cb_remove_color)

for i, c in enumerate(st.session_state.colors):
    label = "Background Color" if i == 0 else f"Color {i}"
    
    if f"color_{i}" not in st.session_state:
        st.session_state[f"color_{i}"] = c["color"]
    if f"threshold_{i}" not in st.session_state:
        st.session_state[f"threshold_{i}"] = c["threshold"]
        
    col_c, col_t = st.columns([1, 2])
    with col_c:
        st.color_picker(
            label,
            key=f"color_{i}",
            on_change=save_params
        )
    with col_t:
        st.slider(
            "Threshold",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            key=f"threshold_{i}",
            label_visibility="visible",
            on_change=save_params
        )
