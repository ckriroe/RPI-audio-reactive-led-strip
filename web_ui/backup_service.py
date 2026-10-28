import os
import io
import json
import zipfile
from datetime import datetime
from web_ui.constants import PRESETS_FILE, STATIC_CONFIG_FILE
from web_ui.storage import load_version, write_static_config_to_disk
from web_ui.migrations import migrate_presets


def generate_backup_filename() -> str:
    ver = load_version()
    if ver:
        ver_clean = ver.strip().lstrip("vV").replace(".", "-")
        v_str = f"_V{ver_clean}"
    else:
        v_str = ""

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    return f"{timestamp}_Backup{v_str}.zip"


def create_backup_zip() -> tuple[bytes, str]:
    try:
        from web_ui.state_sync import save_presets
        import streamlit as st
        if "presets" in st.session_state and len(st.session_state.presets) > 0:
            save_presets()
    except Exception:
        pass

    filename = generate_backup_filename()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(PRESETS_FILE):
            zf.write(PRESETS_FILE, arcname=PRESETS_FILE)
        if os.path.exists(STATIC_CONFIG_FILE):
            zf.write(STATIC_CONFIG_FILE, arcname=STATIC_CONFIG_FILE)

    buf.seek(0)
    return buf.getvalue(), filename


def restore_backup_zip(uploaded_file) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            file_names = zf.namelist()
            presets_member = next((m for m in file_names if os.path.basename(m) == PRESETS_FILE), None)
            static_member = next((m for m in file_names if os.path.basename(m) == STATIC_CONFIG_FILE), None)

            if not presets_member and not static_member:
                return False, f"Weder '{PRESETS_FILE}' noch '{STATIC_CONFIG_FILE}' in der ZIP-Datei gefunden."

            if presets_member:
                presets_raw = zf.read(presets_member).decode("utf-8")
                presets_json = json.loads(presets_raw)
                presets_json = migrate_presets(presets_json)
                with open(PRESETS_FILE, "w", encoding="utf-8") as f:
                    json.dump(presets_json, f, indent=2)

            if static_member:
                static_raw = zf.read(static_member).decode("utf-8")
                static_json = json.loads(static_raw)
                write_static_config_to_disk(static_json)

        return True, ""
    except Exception as e:
        return False, str(e)
