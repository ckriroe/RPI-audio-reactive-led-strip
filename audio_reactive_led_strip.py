import pygame
import numpy as np
import sounddevice as sd
import time
import json
import math
from math import exp
from collections import deque
import threading
from inotify_simple import INotify, flags
from pathlib import Path
import colorsys
from dataclasses import dataclass
import random
import RPi.GPIO as GPIO
import board
import neopixel

class LimitedBuffer:
    def __init__(self, max_size):
        self.buf = deque(maxlen=max_size)

    def add(self, x):
        self.buf.append(x)

    @property
    def items(self):
        return list(self.buf)
    
    @property
    def maxlen(self):
        return self.buf.maxlen

@dataclass
class ColorThreshold:
    color: tuple[int, int, int]
    threshold: float

@dataclass
class LedPixel:
    value: float
    color: tuple[int,int,int]

# --------------------------- Constants --------------------------

# 0 = mock, 1 = led strip
DISPLAY_MODE = 0

# Led constants
LED_COUNT = 278
FPS = 55
MAX_SPEED = 600
BRIGHTNESS = 1.0 # 0.0 – 1.0
EXTERNAL_MODE_RELAY_GPIO = 5
DATA_PIN = board.D12

# Test display settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 200
LED_RADIUS = 2

# Audio processing constants
CONFIG_FILE = "audio_params.json"
FFT_SIZE = 512
SAMPLE_RATE = 44100
BELOW_MIN_FREQ_AMPLITUDE_FUNCTION_FACTOR = -0.03
MAX_FREQ_AMPLITUDE_INCREASE_RATIO = 3
MAX_FREQ_AMPLITUDE_DECREASE_RATIO = 5
MAX_FREQ_AMPLITUDE_TTL_MS = 2000
MAX_FREQ_AMPLITUDE_PROLONGER_THRESHOLD_PERCENT = 0.03
MAX_FREQ_AMPLITUDE_DECAY_RATE = 0.003
PERCENT_DIFF_FROM_MAX_TO_BE_EXTRAORDINARY = 0.30
MIN_SANITIZED_VALUE = 0.01
MAX_BOUNCE_LAYERS = 1

# --------------------------- GLOBAL STATE --------------------------

# Audio state
last_written_uniform_value = -1.0
last_aprox_max_freq = -1.0
last_aprox_max_freq_eval = -1
output_freq_data = True
freq_buffer = []
latest_bass_value = 0
running = True

# Loaded settings
background_color = (0, 0, 0)
use_rainbow = False
effect_origin = 139
speed = 300
min_used_freq = 0
max_used_freq = 180
fade = 0.6
fade_over_time = 0.0
bouncy_wave = False
static_spectrum = True
spectrum_sections = 10
saturate = 0.6
saturate_threshold = 0.3
mean_value_buffer_size = 20
mean_value_threshold = 0.3
last_extra_ordinary_sample_buffer = LimitedBuffer(mean_value_buffer_size)
audio_mode = 0
effect_mode = 0
color_mode = 0
min_freq_amplitude = 0.1
max_freq_amplitude = 10.0
color_increase_factor = 1.0
value_increase_factor = 1.0
value_color_bias = 0.0
get_alpha_from_value = False
color_wave_origin = 139
color_wave_speed = 50
color_wave_size = 100
color_wave_inwards = False
color_overflow = False
noise_amount = 0.00
noise_smoothing = 1.00
color_transition = 0.25
brightness = 1.0
led_count = 10
physical_led_count = 10
gamma = 1.0
color_palette: list[ColorThreshold] = []
effect_repeats = 1

strip = [LedPixel(0.0, (0, 0, 0)) for _ in range(led_count)]
led_noise = [0.0] * led_count
led_strip = None

# --------------------------- FUNCTIONS -----------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Input must be in format #RRGGBB")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def lerp(a, b, t):
    return a + (b - a) * t
    
def lerp_color(c1, c2, t: float):
    return tuple(max(0, int(lerp(a, b, t))) for a, b in zip(c1, c2))

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

last_config_loaded = 0.0
def load_params():
    global effect_origin, speed, min_used_freq, max_used_freq, noise_amount, fade_over_time
    global fade, effect_mode, min_freq_amplitude, noise_smoothing, get_alpha_from_value, bouncy_wave
    global color_increase_factor, background_color, use_rainbow, color_overflow, output_freq_data
    global value_increase_factor, color_mode, value_color_bias, brightness, saturate, led_strip
    global color_palette, color_transition, last_config_loaded, gamma, saturate_threshold
    global color_wave_origin, color_wave_speed, color_wave_size, color_wave_inwards, strip
    global mean_value_buffer_size, mean_value_threshold, max_freq_amplitude, led_count, led_noise
    global static_spectrum, spectrum_sections, audio_mode, effect_repeats, physical_led_count
    
    try:
        if not Path(CONFIG_FILE).exists():
            return

        now = int(time.time() * 1000)
        if (now - last_config_loaded < 0.3):
            return
        
        last_config_loaded = now
        time.sleep(0.15)
        with open(CONFIG_FILE) as f:
            data = json.load(f)

        new_effect_repeats = data.get("effectRepeats", effect_repeats)
        new_physical_led_count = data.get("ledCount", led_count)
        if physical_led_count != new_physical_led_count or new_effect_repeats != effect_repeats:
            effect_repeats = new_effect_repeats
            physical_led_count = new_physical_led_count
            led_count = max(2, int(new_physical_led_count / effect_repeats))
            led_noise = [0.0] * led_count
            strip = [LedPixel(0.0, (0, 0, 0)) for _ in range(led_count)]
            
            if led_strip is not None:
                led_strip.fill((0, 0, 0))
                led_strip.show()
                led_strip.deinit()

            led_strip = neopixel.NeoPixel(
                DATA_PIN,
                new_physical_led_count,
                brightness=BRIGHTNESS,
                auto_write=False,
                pixel_order=neopixel.GRB
            )
        
        use_rainbow = data.get("useRainbow", use_rainbow)
        effect_origin = min(led_count - 1, int(data.get("effectOrigin", effect_origin) / effect_repeats))
        speed = data.get("speed", speed)
        new_min_used_freq = data.get("minFreq", min_used_freq)
        new_max_used_freq = data.get("maxFreq", max_used_freq)

        if new_min_used_freq > new_max_used_freq:
            new_max_used_freq = new_min_used_freq
            
        if new_max_used_freq != max_used_freq:
            max_used_freq = new_max_used_freq
            output_freq_data = True

        if new_min_used_freq != min_used_freq:
            min_used_freq = new_min_used_freq
            output_freq_data = True

        fade = data.get("fade", fade)
        fade_over_time = data.get("fadeOverTime", fade_over_time) / 1000.0
        bouncy_wave = data.get("bouncyWave", bouncy_wave)
        static_spectrum = data.get("staticSpectrum", static_spectrum)
        spectrum_sections = data.get("spectrumSections", spectrum_sections)
        saturate = data.get("saturate", saturate)
        saturate_threshold = data.get("saturateThreshold", saturate_threshold)
        mean_value_buffer_size = data.get("meanValueBufferSize", mean_value_buffer_size)
        mean_value_threshold = data.get("meanValueThreshold", mean_value_threshold)
        audio_mode = data.get("audioMode", audio_mode)
        effect_mode = data.get("effectMode", effect_mode)
        color_mode = data.get("colorMode", color_mode)
        min_freq_amplitude = data.get("minFreqAmplitude", min_freq_amplitude)
        max_freq_amplitude = data.get("maxFreqAmplitude", max_freq_amplitude)
        color_increase_factor = data.get("colorIncreaseFactor", color_increase_factor)
        value_increase_factor = data.get("valueIncreaseFactor", value_increase_factor)
        value_color_bias = data.get("valueColorBias", value_color_bias)
        get_alpha_from_value = data.get("getAlphaFromValue", get_alpha_from_value)
        color_overflow = data.get("colorOverflow", color_overflow)
        color_transition = data.get("colorTransition", color_transition)
        
        color_wave_origin = min(led_count - 1, int(data.get("colorWaveOrigin", color_wave_origin) / effect_repeats))
        color_wave_speed = data.get("colorWaveSpeed", color_wave_speed)
        color_wave_size = data.get("colorWaveSize", color_wave_size)
        color_wave_inwards = data.get("colorWaveInwards", color_wave_inwards)
        noise_amount = data.get("noiseAmount", noise_amount)
        noise_smoothing = data.get("noiseSmoothing", noise_smoothing)
        brightness = data.get("brightness", brightness)
        gamma = data.get("gamma", gamma)
        
        color_palette = []

        raw_colors = data.get("colors", [])

        last_threshold = 0.0

        for entry in raw_colors:
            try:
                color_hex = entry.get("color", "#000000")
                threshold = float(entry.get("threshold", 1.0))

                if threshold < last_threshold:
                    continue;

                threshold = max(0.0, min(1.0, threshold))

                color_palette.append(
                    ColorThreshold(
                        color=hex_to_rgb(color_hex),
                        threshold=threshold
                    )
                )

                last_threshold = threshold

            except Exception as e:
                print("Invalid color entry skipped:", e)

        if color_palette and len(color_palette) > 1:
            color_palette[-1].threshold = 1.0

        print("Config updated")

    except Exception as e:
        print("Error reading config:", e)

def value_to_rainbow_color(value: float):
    background_entry = color_palette[0]
    transition_area_size = 0.0
    
    if background_entry.threshold > 0.5:
        transition_area_size = (1.0 - background_entry.threshold) * color_transition
    else:
        transition_area_size = background_entry.threshold * color_transition
        
    if value < background_entry.threshold - transition_area_size or background_entry.threshold == 1.0:
        return background_entry.color
    
    t = (value - background_entry.threshold) / (1.0 - background_entry.threshold)
    hue = t * 0.75
    if hue < 0.0:
        hue = 0.0
    
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    color = (
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )
    
    if value < background_entry.threshold:
        if value < background_entry.threshold - transition_area_size or transition_area_size == 0:
            return background_entry.color

        t = (background_entry.threshold - value) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(color, background_entry.color, 0.5), background_entry.color, t)
    else:
        if value > background_entry.threshold + transition_area_size or transition_area_size == 0:
            return color
        
        t = (value - background_entry.threshold) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(background_entry.color, color, 0.5), color, t)

def value_to_color(value: float) -> tuple[int, int, int]:
    value *= color_increase_factor

    if value > 1.0 and color_overflow:
        if (int(value) % 2 == 0):
            value = value % 1.0
        else:
            value = 1.0 - (value % 1.0)    
    
    if value > 1.0:
        value = 1.0
    
    if len(color_palette) == 0:
        return (0, 0, 0)
    
    if use_rainbow:
        return value_to_rainbow_color(value)  
    
    if len(color_palette) < 2:
        return color_palette[0].color

    palette_for_color_idx = 0
    color_section_end = color_palette[0].threshold
    curr_color = color_palette[0].color
    for i, entry in enumerate(color_palette):
        if value < entry.threshold or (value == 1.0 and value <= entry.threshold):
            palette_for_color_idx = i
            color_section_end = entry.threshold
            curr_color = entry.color
            break
        
    color_section_start = 0.0
    if palette_for_color_idx != 0:
        color_section_start = color_palette[palette_for_color_idx - 1].threshold
        
    color_section_size = color_section_end - color_section_start
    transition_area_size = 0.0
    is_above_half_way = value >= color_section_size / 2.0 + color_section_start
    
    if is_above_half_way:
        if palette_for_color_idx == len(color_palette) - 1:
            return color_palette[palette_for_color_idx].color
        
        next_section_end = color_palette[palette_for_color_idx + 1].threshold
        next_section_size = next_section_end - color_section_end
        if next_section_size > color_section_size:
            transition_area_size = color_section_size * color_transition
        else:        
            transition_area_size = next_section_size * color_transition
    else:
        if palette_for_color_idx == 0:
            return color_palette[palette_for_color_idx].color
        
        prev_section_start = 0
        if (palette_for_color_idx > 1):
            prev_section_start = color_palette[palette_for_color_idx - 2].threshold
        
        prev_section_size = color_section_start - prev_section_start
        if prev_section_size > color_section_size:
            transition_area_size = color_section_size * color_transition
        else:        
            transition_area_size = prev_section_size * color_transition
    
    if is_above_half_way:
        if value < color_section_end - transition_area_size or transition_area_size == 0:
            return color_palette[palette_for_color_idx].color

        t = (color_section_end - value) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(color_palette[palette_for_color_idx + 1].color, curr_color, 0.5), curr_color, t)
    else:
        if value > color_section_start + transition_area_size or transition_area_size == 0:
            return color_palette[palette_for_color_idx].color
        
        t = (value - color_section_start) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(color_palette[palette_for_color_idx - 1].color, curr_color, 0.5), curr_color, t)

def non_audio_value_to_color(value: float, audio_value: float) -> tuple[int, int, int]:
    audio_color = value_to_color(audio_value)
    
    prev_val_theshold = color_palette[0].threshold
    color_palette[0].threshold = 0    
    color = lerp_color(value_to_color(value), audio_color, value_color_bias)
    color_palette[0].threshold = prev_val_theshold

    if not get_alpha_from_value:
        return color

    background_entry = color_palette[0]
    transition_area_size = 0.0
    
    if background_entry.threshold > 0.5:
        transition_area_size = (1.0 - background_entry.threshold) * color_transition
    else:
        transition_area_size = background_entry.threshold * color_transition
        
    if audio_value < background_entry.threshold - transition_area_size or background_entry.threshold == 1.0:
        return background_entry.color
    
    if audio_value < background_entry.threshold:
        if audio_value < background_entry.threshold - transition_area_size or transition_area_size == 0:
            return background_entry.color

        t = (background_entry.threshold - audio_value) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(color, background_entry.color, 0.5), background_entry.color, t)
    else:
        if audio_value > background_entry.threshold + transition_area_size or transition_area_size == 0:
            return color
        
        t = (audio_value - background_entry.threshold) / transition_area_size
        t = max(0.0, min(1.0, t))
        return lerp_color(lerp_color(background_entry.color, color, 0.5), color, t)

wave_distance_accumulator = 0.0
prev_wave_strip = []
def get_wave_mode_led_state(prev_strip: list[LedPixel], new_value: float):
    global wave_distance_accumulator, prev_wave_strip

    bounce_layers = MAX_BOUNCE_LAYERS
    if bouncy_wave == False:
        bounce_layers = 0

    n = len(prev_strip) * (1 + bounce_layers * 2)
    center = effect_origin + len(prev_strip) * bounce_layers
    wave_distance_accumulator += speed / FPS
    steps = int(wave_distance_accumulator)
    wave_distance_accumulator -= steps

    if steps == 0 or center >= n:
        return prev_strip
   
    if len(prev_wave_strip) != n:
        prev_wave_strip = [LedPixel(0.0, (0, 0, 0)) for _ in range(n)]
        
    new_strip = [LedPixel(p.value, (0, 0, 0)) for p in prev_wave_strip]
    for i in range(0, center + 1):
        src_index = i
        dst_index = i - steps
        if dst_index < 0 or src_index > n - 1:
            continue
        
        new_strip[dst_index].value = prev_wave_strip[src_index].value

        if src_index != center:
            new_strip[src_index].value = 0.0

    for i in range(n - 1, center - 1, -1):
        src_index = i
        dst_index = i + steps
        if dst_index > n - 1 or src_index < 0:
            continue

        new_strip[dst_index].value = prev_wave_strip[src_index].value

        if src_index != center:
            new_strip[src_index].value = 0.0

    for direction in (-1, 1):
        idx = center + direction

        if idx < 0 or idx >= n:
            continue

        while 0 <= idx < n and new_strip[idx].value == 0.0:
            idx += direction

        if not (0 <= idx < n):
            continue

        target_idx = idx
        target_value = new_strip[target_idx].value

        gap = abs(target_idx - center) - 1
        fill = min(gap, steps)

        for i in range(1, fill + 1):
            pos = center + direction * i
            t = i / (gap + 1)
            v = lerp(new_value, target_value, t)
            new_strip[pos] = LedPixel(v, (0, 0, 0))

    new_strip[center] = LedPixel(new_value, (0, 0, 0))
    new_strip = fade_center_based_strip(new_strip, center)
    prev_wave_strip = new_strip
    if bounce_layers == 0:
        return new_strip
    
    bll = len(prev_strip)
    bounced_new_strip = [LedPixel(0.0, (0, 0, 0)) for _ in range(bll)]
    
    for i in range(len(bounced_new_strip)):
        base_val = new_strip[i + bounce_layers * bll].value
        for bl in range(bounce_layers):
            is_bl_reverse = bl % 2 == 0
            
            left_bl_val_index = ((bounce_layers - 1) - bl) * bll
            if is_bl_reverse:
                left_bl_val_index = (left_bl_val_index + (bll - 1)) - i
            else:
                left_bl_val_index += i
            
            right_bl_val_index = (bounce_layers + bl + 1) * bll
            if is_bl_reverse:
                right_bl_val_index = (right_bl_val_index + (bll - 1)) - i
            else:
                right_bl_val_index += i
               
            left_bl_val = new_strip[left_bl_val_index].value
            right_bl_val = new_strip[right_bl_val_index].value
            if left_bl_val > base_val:
                base_val = left_bl_val
                
            if right_bl_val > base_val:
                base_val = right_bl_val                
        
        bounced_new_strip[i].value = base_val
    
    return bounced_new_strip

def get_pulsate_led_state(prev_strip: list[LedPixel], new_value: float):
    return [LedPixel(new_value, (0, 0, 0)) for p in prev_strip]

def get_line_led_state(prev_strip: list[LedPixel], new_value: float):
    n = len(prev_strip)
    new_strip = []

    center = effect_origin
    left_dist = center
    right_dist = (n - 1) - center
    left_extent = int(left_dist * new_value)
    right_extent = int(right_dist * new_value)

    start = center - left_extent
    end = center + right_extent

    for i in range(n):
        if start <= i <= end:
            new_strip.append(LedPixel(new_value, (0, 0, 0)))
        else:
            new_strip.append(LedPixel(0.0, background_color))

    return new_strip

def get_spectrum_led_state(prev_strip: list[LedPixel]):
    n = len(prev_strip)

    center = effect_origin
    bins = freq_buffer
    bin_count = len(bins)

    if bin_count == 0:
        return [LedPixel(0.0, background_color) for _ in range(n)]
    
    max_dist = max(center, (n - 1) - center)
    section_length = len(bins) / spectrum_sections
    new_strip = []

    for i in range(n):
        dist = abs(i - center)

        if bin_count == 1:
            amp = float(bins[0])
        elif static_spectrum:
            bin_pos = (dist / max_dist) * (bin_count - 1)
            section = int(bin_pos / section_length)
            start_bin = max(0, int(section * len(bins) // spectrum_sections))
            end_bin = min(len(bins) - 1, int((section + 1) * len(bins) // spectrum_sections))
            if start_bin == end_bin:
                amp = bins[start_bin]
            else:
                amp = max(bins[start_bin:end_bin])
        else:
            bin_pos = (dist / max_dist) * (bin_count - 1)
            b0 = int(bin_pos)
            b1 = min(b0 + 1, bin_count - 1)
            t = bin_pos - b0
            amp = (1 - t) * float(bins[b0]) + t * float(bins[b1])

        if amp > max_freq_amplitude:
            amp = max_freq_amplitude
            
        if amp < min_freq_amplitude:
            amp = 0.0
            
        amp *= value_increase_factor
        
        if max_freq_amplitude == 0.0:
            value = 0.0
        else:
            value = amp / max_freq_amplitude

        value = max(0.0, min(1.0, value))
        
        if value > prev_strip[i].value and value > saturate_threshold:
            value = lerp(prev_strip[i].value, value, saturate)
        else:
            value = prev_strip[i].value * fade

        new_strip.append(LedPixel(value, (0, 0, 0)))

    return new_strip

def get_random_burst_led_state(prev_strip: list[LedPixel], value: float):
    n = len(prev_strip)
    new_strip = []

    for p in prev_strip:
        faded_value = p.value * fade
        new_strip.append(LedPixel(faded_value, (0, 0, 0)))

    intensity_radius = int(value_increase_factor)
    probability_factor = (speed / MAX_SPEED) * 0.01
    effective_prob = min(1.0, value * probability_factor)

    for i in range(n):
        if random.random() < effective_prob:
            new_strip[i] = LedPixel(value, value_to_color(value))

            for offset in range(1, intensity_radius + 1):
                for neighbor in (i - offset, i + offset):
                    if 0 <= neighbor < n:
                        t = offset / (intensity_radius + 1)
                        interp_val = value * (1.0 - t)
                        new_strip[neighbor] = LedPixel(interp_val, (0, 0, 0))

    return new_strip

def get_static_asc_led_state(prev_strip: list[LedPixel]):
    new_strip = []
    
    for i, led in enumerate(prev_strip):
        percentage = i / (len(prev_strip) - 1)
        new_strip.append(LedPixel(percentage, (0, 0, 0)))
        
    return new_strip
    
def get_static_led_state(prev_strip: list[LedPixel]):
    new_strip = []
    
    for i, led in enumerate(prev_strip):
        new_strip.append(LedPixel(max(0.0, min(1.0, 1.0 * value_increase_factor)), (0, 0, 0)))
        
    return new_strip
    
def fade_center_based_strip(strip_to_fade: list[LedPixel], center = effect_origin):
    for i in range(len(strip_to_fade)):
        dist_to_center = abs(center - i)
        strip_to_fade[i].value *= min(10.0, max(0.0, (1.0 - (dist_to_center * fade_over_time))))
    return strip_to_fade

def get_led_state(prev_strip: list[LedPixel], new_value: float):
    if effect_mode == 0:
        return get_pulsate_led_state(prev_strip, new_value)
    elif effect_mode == 1:
        return get_line_led_state(prev_strip, new_value)
    elif effect_mode == 2:
        return get_wave_mode_led_state(prev_strip, new_value)
    elif effect_mode == 3:
        return get_spectrum_led_state(prev_strip)
    elif effect_mode == 4:
        return get_random_burst_led_state(prev_strip, new_value)
    elif effect_mode == 5:
        return get_static_asc_led_state(prev_strip)
    elif effect_mode == 6:
        return get_static_led_state(prev_strip)
    else:
        return prev_strip

def render_led_state_pygame(new_strip, screen_to_draw):
    screen_to_draw.fill((1, 1, 1))
    spacing = int(SCREEN_WIDTH / physical_led_count)
    
    cntr = 0
    for r in range(effect_repeats):       
        for i in range(len(new_strip)):
            if cntr >= physical_led_count:
                break;
            
            led = new_strip[i]
            
            x = cntr * spacing + spacing / 2
            y = SCREEN_HEIGHT // 2
            pygame.draw.circle(
                screen_to_draw,
                led.color,
                (int(x), int(y)),
                LED_RADIUS
            )
            cntr += 1
        
        if cntr >= physical_led_count:
            break
        
    pygame.display.flip()

def write_pixels(strip_to_display: list[LedPixel]):
    if led_strip is None:
        return
    
    cntr = 0
    for r in range(effect_repeats):
        for i in range(len(strip_to_display)):
            if cntr >= physical_led_count or cntr >= len(led_strip):
                break
                
            led_strip[cntr] = strip_to_display[i].color
            cntr += 1
            
        if cntr >= physical_led_count:
            break

    led_strip.show()

def apply_smooth_noise(
    value: float,
    idx: int
) -> float:
    global led_noise
    if noise_amount == 0.0 or value == 0.0:
        return value

    target = random.uniform(-noise_amount, noise_amount)
    if idx >= len(led_noise):
        return value
    
    led_noise[idx] = (
        led_noise[idx] * noise_smoothing +
        target * (1.0 - noise_smoothing)
    )

    v = value + led_noise[idx]
    return max(0.0, min(1.0, v))

def sanitize_values(strip: list[LedPixel]):
    for led in strip:
        if led.value < MIN_SANITIZED_VALUE:
            led.value = 0

def clear_strip(strip_to_color: list[LedPixel]):
    for i, led in enumerate(strip_to_color):
        led.color = (0, 0, 0)

def color_strip_by_value(strip_to_color: list[LedPixel]):
    for i, led in enumerate(strip_to_color):
        led.color = value_to_color(apply_smooth_noise(led.value, i))

def color_strip_by_index(strip_to_color: list[LedPixel]):
    n = len(strip_to_color)

    if n <= 1 or len(color_palette) == 0:
        return

    last_idx = n - 1

    for i, led in enumerate(strip_to_color):
        if led.value > 0.01:
            t = i / last_idx         
            led.color = non_audio_value_to_color(apply_smooth_noise(t, i), apply_smooth_noise(led.value, i))
        else:
            led.color = color_palette[0].color
            
def color_strip_by_distance_to_center(strip_to_color: list[LedPixel]):
    n = len(strip_to_color)

    if n <= 1 or len(color_palette) == 0:
        return

    center = effect_origin
    max_dist = max(center, (n - 1) - center)
    if max_dist <= 0:
        max_dist = 1

    for i, led in enumerate(strip_to_color):
        if led.value > 0.01:
            dist = abs(i - center)
            t = dist / max_dist
            led.color = non_audio_value_to_color(apply_smooth_noise(t, i), apply_smooth_noise(led.value, i))
        else:
            led.color = color_palette[0].color    

def color_strip_by_distance_to_edge(strip_to_color: list[LedPixel]):
    n = len(strip_to_color)

    if n <= 1 or len(color_palette) == 0:
        return

    max_dist = (n - 1) / 2
    if max_dist <= 0:
        max_dist = 1

    for i, led in enumerate(strip_to_color):
        if led.value > 0.01:
            dist = min(i, (n - 1) - i)
            t = dist / max_dist
            led.color = non_audio_value_to_color(apply_smooth_noise(t, i), apply_smooth_noise(led.value, i))
        else:
            led.color = color_palette[0].color

def wave_envelope(x: float) -> float:
    if x < 0.5:
        return x * 2.0
    return (1.0 - x) * 2.0

color_wave_phase = 0.0
def color_strip_wave(strip_to_color: list[LedPixel]):
    global color_wave_phase

    n = len(strip_to_color)
    if n == 0 or len(color_palette) == 0:
        return

    direction = -1
    if color_wave_inwards:
        direction = 1

    color_wave_phase += direction * color_wave_speed / FPS
    if color_wave_phase >= color_wave_size:
        color_wave_phase -= color_wave_size

    half_wave = color_wave_size / 2.0

    for i, led in enumerate(strip_to_color):
        if led.value <= 0.01:
            led.color = color_palette[0].color
            continue

        dist = abs(i - color_wave_origin)
        wave_pos = (dist + color_wave_phase) % color_wave_size
        t = wave_pos / color_wave_size
        wave_value = wave_envelope(t)
        led.color = non_audio_value_to_color(apply_smooth_noise(wave_value, i), apply_smooth_noise(led.value, i))    

def color_strip_by_islands(strip_to_color: list[LedPixel]):
    n = len(strip_to_color)

    if n == 0 or len(color_palette) == 0:
        return

    i = 0
    while i < n:
        if strip_to_color[i].value <= 0.0:
            strip_to_color[i].color = color_palette[0].color
            i += 1
            continue

        start = i
        while i < n and strip_to_color[i].value > 0.0:
            i += 1
        end = i - 1

        length = end - start + 1
        if length == 1:
            led = strip_to_color[start]
            led.color = non_audio_value_to_color(apply_smooth_noise(1.0, start), apply_smooth_noise(led.value, start))
            continue

        mid = (length - 1) / 2.0

        for j in range(length):
            idx = start + j
            led = strip_to_color[idx]
            dist = abs(j - mid) / mid if mid != 0 else 0.0
            t = 1.0 - dist
            t = max(0.0, min(1.0, t))          
            led.color = non_audio_value_to_color(apply_smooth_noise(t, idx), apply_smooth_noise(led.value, idx))

def color_strip(strip_to_color: list[LedPixel]):
    if color_mode == 0:
        color_strip_by_value(strip_to_color)
    elif color_mode == 1:
        color_strip_by_index(strip_to_color)
    elif color_mode == 2:
        color_strip_by_distance_to_center(strip_to_color)
    elif color_mode == 3:
        color_strip_by_distance_to_edge(strip_to_color)
    elif color_mode == 4:
        color_strip_wave(strip_to_color)
    elif color_mode == 5:
        color_strip_by_islands(strip_to_color)

def process_audio_data(indata):
    global last_aprox_max_freq
    global last_aprox_max_freq_eval
    global output_freq_data
    global freq_buffer
    global last_extra_ordinary_sample_buffer

    audio = indata[:, 0].astype(np.float32)
    fft = np.fft.rfft(audio, n=FFT_SIZE)
    mags = np.abs(fft)
    freq_per_bin = SAMPLE_RATE / FFT_SIZE
    min_bin = int(round(min_used_freq / freq_per_bin))
    max_bin = int(round(max_used_freq / freq_per_bin)) + 1
    freq_buffer = mags[min_bin:max_bin]

    if output_freq_data:
        print(f"Actual frequency range: {min_bin * freq_per_bin}hz - {(max_bin - 1) * freq_per_bin}hz")
        output_freq_data = False
    
    max_freq = np.max(freq_buffer)
    
    if audio_mode == 1:
        diff = max_freq_amplitude - min_freq_amplitude
        if diff <= 0.0:
            return 0.0
                
        return max(0.0, min(1.0, (max_freq - min_freq_amplitude) / diff))
    
    current_time_ms = int(time.time() * 1000)

    if max_freq > last_aprox_max_freq:
        last_aprox_max_freq = (last_aprox_max_freq + max_freq * (MAX_FREQ_AMPLITUDE_INCREASE_RATIO - 1)) / MAX_FREQ_AMPLITUDE_INCREASE_RATIO
        last_aprox_max_freq_eval = current_time_ms
    elif max_freq > last_aprox_max_freq - last_aprox_max_freq * MAX_FREQ_AMPLITUDE_PROLONGER_THRESHOLD_PERCENT:
        last_aprox_max_freq = (last_aprox_max_freq * (MAX_FREQ_AMPLITUDE_DECREASE_RATIO - 1) + max_freq) / MAX_FREQ_AMPLITUDE_DECREASE_RATIO
        last_aprox_max_freq_eval = current_time_ms
        max_freq = last_aprox_max_freq
    elif current_time_ms - last_aprox_max_freq_eval > MAX_FREQ_AMPLITUDE_TTL_MS:
        last_aprox_max_freq *= (1.0 - MAX_FREQ_AMPLITUDE_DECAY_RATE)

    if last_extra_ordinary_sample_buffer.maxlen != mean_value_buffer_size:
        last_extra_ordinary_sample_buffer = LimitedBuffer(mean_value_buffer_size)

    if max_freq < last_aprox_max_freq - last_aprox_max_freq * mean_value_threshold or not last_extra_ordinary_sample_buffer.items:
        last_extra_ordinary_sample_buffer.add(max_freq)

    avg = np.mean(last_extra_ordinary_sample_buffer.items) if last_extra_ordinary_sample_buffer.items else 0

    if max_freq > max_freq_amplitude:
        max_freq = max_freq_amplitude

    if max_freq > min_freq_amplitude:
        adjusted_freq_value = max_freq
    else:
        scale_factor = BELOW_MIN_FREQ_AMPLITUDE_FUNCTION_FACTOR
        adjusted_freq_value = min_freq_amplitude - (1.0 / scale_factor) + (1.0 / scale_factor) * exp(scale_factor * (max_freq - min_freq_amplitude))
        adjusted_freq_value = max(0.0, adjusted_freq_value)
    
    uniform_value = max(0, (adjusted_freq_value - avg) / last_aprox_max_freq if last_aprox_max_freq != 0 else 0)
    return uniform_value

def audio_thread():
    global latest_bass_value
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32', blocksize=FFT_SIZE) as stream:
        while running:
            indata, _ = stream.read(FFT_SIZE)
            latest_bass_value = process_audio_data(indata)

def config_thread():
    load_params()
    while True:
        for ev in inotify.read():
            if flags.MODIFY in flags.from_mask(ev.mask):
                load_params()             

def render_led_strip(strip, screen):
    #if DISPLAY_MODE == 0:
    # render_led_state_pygame(strip, screen)
    #else:
    write_pixels(strip)

def color_correct_strip(strip_to_correct: list[LedPixel]):
    for i, led in enumerate(strip_to_correct):
        led.color = (min(255.0, gamma_correct(led.color[0] * brightness)), min(255.0, gamma_correct(led.color[1] * brightness)), min(255.0, gamma_correct(led.color[2] * brightness)))
        
def gamma_correct(value):
    return int((value / 255.0) ** gamma * 255.0 + 0.5)

# --------------------------- Main sector --------------------------

clock = pygame.time.Clock()

if DISPLAY_MODE == 0:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LED Strip Wave Visualizer")

inotify = INotify()
watch_flags = flags.MODIFY
wd = inotify.add_watch(CONFIG_FILE, watch_flags)

audio_thread = threading.Thread(target=audio_thread, daemon=True)
audio_thread.start()

config_thread = threading.Thread(target=config_thread, daemon=True)
config_thread.start()

current_led_value = 0.0
GPIO.setmode(GPIO.BCM)
GPIO.setup(EXTERNAL_MODE_RELAY_GPIO, GPIO.OUT)
is_in_external_mode = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if latest_bass_value > current_led_value and latest_bass_value > saturate_threshold:
        current_led_value = lerp(current_led_value, latest_bass_value, saturate)
    else:
        current_led_value *= fade
   
    if effect_mode == 7 and is_in_external_mode == False:
        is_in_external_mode = True
        clear_strip(strip)
        render_led_strip(strip, screen)
        time.sleep(1)
        GPIO.output(EXTERNAL_MODE_RELAY_GPIO, GPIO.LOW)
    elif effect_mode != 7 and is_in_external_mode != False:
        is_in_external_mode = False
        GPIO.output(EXTERNAL_MODE_RELAY_GPIO, GPIO.HIGH)
        
    if is_in_external_mode == True:
        time.sleep(1)
        continue
    
    strip = get_led_state(
        strip,
        current_led_value * value_increase_factor
    )
    
    sanitize_values(strip)
    color_strip(strip)
    color_correct_strip(strip)

    clock.tick(FPS)
    render_led_strip(strip, screen)

if DISPLAY_MODE == 0:
    pygame.quit()

running = False
audio_thread.join()
config_thread.join()
GPIO.cleanup()
