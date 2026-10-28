import streamlit as st
import streamlit.components.v1 as components

SLIDER_FIX_JS = """
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

CUSTOM_CSS = """
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
