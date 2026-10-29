import streamlit as st
import streamlit.components.v1 as components

SLIDER_FIX_JS = """
<script>
(function() {
    try {
        const parentWindow = window.parent;
        if (!parentWindow || !parentWindow.document) return;
        const parentDoc = parentWindow.document;

        const eventsToIntercept = ['pointerdown', 'touchstart', 'mousedown', 'click'];

        // If previous handlers exist, remove them first so reruns always update cleanly
        if (parentWindow.__slider_fix_handlers) {
            eventsToIntercept.forEach(eventType => {
                if (parentWindow.__slider_fix_handlers[eventType]) {
                    parentDoc.removeEventListener(eventType, parentWindow.__slider_fix_handlers[eventType], { capture: true, passive: true });
                }
            });
        }
        parentWindow.__slider_fix_handlers = {};

        function isMobile(e) {
            if (e && (e.pointerType === 'touch' || (e.type && e.type.startsWith('touch')))) {
                return true;
            }
            if (parentWindow.matchMedia && (parentWindow.matchMedia('(pointer: coarse)').matches || parentWindow.matchMedia('(hover: none)').matches)) {
                return true;
            }
            const hasTouch = 'ontouchstart' in parentWindow || (parentWindow.navigator && parentWindow.navigator.maxTouchPoints > 0);
            return hasTouch;
        }

        eventsToIntercept.forEach(eventType => {
            const handler = function(e) {
                if (!isMobile(e)) return;

                // Check if target is inside a Streamlit slider
                const slider = e.target && e.target.closest && e.target.closest('[data-testid="stSlider"], .stSlider, [class*="efbyxod0"], [class*="efbyxod1"], [class*="react-aria-Slider"], [data-baseweb="slider"]');
                if (!slider) return;

                // Check if target is on or inside the knob (efbyxod3 / 1v3hvtd)
                const thumb = e.target.closest && e.target.closest('[class*="efbyxod3"], [class*="1v3hvtd"], [class*="SliderThumb"], [role="slider"], input[type="range"], [data-testid="stSliderThumbValue"]');
                if (thumb) {
                    // Touching the knob: allow dragging!
                    return;
                }

                // Touching the track or outside the knob: block it from changing the slider
                e.stopPropagation();
                e.stopImmediatePropagation();
            };

            parentWindow.__slider_fix_handlers[eventType] = handler;
            parentDoc.addEventListener(eventType, handler, { capture: true, passive: true });
        });
    } catch (err) {
        console.error('SLIDER_FIX_JS error:', err);
    }
})();
</script>
"""

CUSTOM_CSS = """
<style>
/* Prevent inner scroll containers */
div[data-testid="stHorizontalBlock"],
div[data-testid="stColumn"] {
    overflow: visible !important;
}

/* Mobile slider touch protection: prevent accidental track jumps while scrolling */
@media (pointer: coarse), (hover: none) {
    div[data-testid="stSlider"] [class*="efbyxod2"],
    div[data-testid="stSlider"] [class*="react-aria-SliderTrack"],
    div[data-testid="stSlider"] [class*="SliderTrack"] {
        pointer-events: none !important;
        touch-action: pan-y !important;
    }
    div[data-testid="stSlider"] [class*="efbyxod3"],
    div[data-testid="stSlider"] [class*="1v3hvtd"],
    div[data-testid="stSlider"] [class*="react-aria-SliderThumb"],
    div[data-testid="stSlider"] [class*="SliderThumb"],
    div[data-testid="stSlider"] input[type="range"] {
        pointer-events: auto !important;
        touch-action: none !important;
    }
}

/* Preset list buttons: clickable button appearance, left-aligned, full width, 38px height */
div[class*="st-key-btn_pre_"] button {
    justify-content: flex-start !important;
    text-align: left !important;
    min-height: 38px !important;
    height: 38px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    margin: 0 !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    transition: all 0.15s ease-in-out !important;
}
div[class*="st-key-btn_pre_"] button:hover {
    background-color: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    cursor: pointer !important;
}
div[class*="st-key-btn_pre_"] button:active {
    transform: scale(0.99) !important;
}
div[class*="st-key-btn_pre_"] button p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    font-size: inherit !important;
}

/* Active preset button highlight */
div[class*="st-key-btn_pre_active_"] button {
    background-color: rgba(46, 204, 113, 0.16) !important;
    border: 1px solid rgba(46, 204, 113, 0.6) !important;
    border-left: 5px solid #2ecc71 !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
div[class*="st-key-btn_pre_active_"] button:hover {
    background-color: rgba(46, 204, 113, 0.24) !important;
    border-color: #2ecc71 !important;
}

/* Action buttons: 38px square buttons */
div[class*="st-key-mv_up_"] button,
div[class*="st-key-mv_down_"] button,
div[class*="st-key-rm_col_"] button,
div[class*="st-key-btn_shutdown"] button,
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

/* Disable margin-right for st-emotion-cache-hqqroh and all its dynamic counterparts */
.st-emotion-cache-hqqroh,
[class*="st-emotion-cache-hqqroh"],
[class*="hqqroh"],
button[data-testid="stPopoverButton"] > div,
button[data-testid="stPopoverButton"] div[class*="st-emotion-cache-"],
button[data-testid="stPopoverButton"] [class*="e1i31rv60"],
button[data-testid="stPopoverButton"] [class*="ecfxx9g1"] {
    margin-right: 0 !important;
}

/* Remove min-width (8rem) from color picker container and columns */
.st-emotion-cache-lhethk,
[class*="st-emotion-cache-lhethk"],
[class*="lhethk"],
.stColorPicker,
div[data-testid="stColorPicker"],
div[data-testid="stColorPicker"] > div,
div[data-testid="stHorizontalBlock"]:has([data-testid="stColorPicker"]) > div[data-testid="stColumn"],
div[data-testid="stColumn"]:has([data-testid="stColorPicker"]),
div[data-testid="stColumn"]:has(.stColorPicker) {
    min-width: 0 !important;
}

/* Preset row layout: preset button takes all free space, popover takes 38px */
div[data-testid="stHorizontalBlock"]:has([class*="pop_pre_"]) {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    width: 100% !important;
    margin-bottom: 0.25rem !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="pop_pre_"]) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="pop_pre_"]) > div[data-testid="stColumn"]:last-child,
div[data-testid="stColumn"]:has([class*="pop_pre_"]) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 38px !important;
}

/* Shutdown button styling */
div[class*="st-key-btn_shutdown"] button {
    font-size: 1.25rem !important;
    color: #ff4b4b !important;
    border-color: rgba(255, 75, 75, 0.4) !important;
}
div[class*="st-key-btn_shutdown"] button:hover {
    color: #ffffff !important;
    background-color: #ff4b4b !important;
    border-color: #ff4b4b !important;
}

/* Header row: title takes all free space, shutdown button takes 38px aligned to right */
div[data-testid="stHorizontalBlock"]:has([class*="st-key-btn_shutdown"]) {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    width: 100% !important;
    margin-bottom: 0.5rem !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-btn_shutdown"]) > div[data-testid="stColumn"]:first-child {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-btn_shutdown"]) > div[data-testid="stColumn"]:not(:first-child),
div[data-testid="stColumn"]:has([class*="st-key-btn_shutdown"]) {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 38px !important;
}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-btn_shutdown"]) h1 {
    margin: 0 !important;
    padding: 0 !important;
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
"""


def inject_custom_styles():
    components.html(SLIDER_FIX_JS, height=0, width=0)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
