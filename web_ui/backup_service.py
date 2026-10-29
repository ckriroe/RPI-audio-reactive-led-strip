import os
import io
import json
import zipfile
from datetime import datetime
from web_ui.constants import PRESETS_FILE, STATIC_CONFIG_FILE, BACKUPS_DIR
from web_ui.storage import load_version, write_static_config_to_disk, atomic_write_json
from web_ui.migrations import migrate_presets


def get_backups_dir() -> str:
    """Returns the local backups directory path, creating it if it doesn't exist."""
    if not os.path.exists(BACKUPS_DIR):
        try:
            os.makedirs(BACKUPS_DIR, exist_ok=True)
        except Exception:
            pass
    return BACKUPS_DIR


def generate_backup_filename() -> str:
    ver = load_version()
    if ver:
        ver_clean = ver.strip().lstrip("vV").replace(".", "-")
        v_str = f"_V{ver_clean}"
    else:
        v_str = ""

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_Backup{v_str}.zip"


def save_backup_locally(filename: str, zip_bytes: bytes) -> str:
    """Saves the backup zip bytes to the local backups folder."""
    b_dir = get_backups_dir()
    clean_filename = os.path.basename(filename)
    target_path = os.path.join(b_dir, clean_filename)
    with open(target_path, "wb") as f:
        f.write(zip_bytes)
    return target_path


def create_backup_zip() -> tuple[bytes, str]:
    """Creates a zip archive of current settings, persists a local copy in backups/, and returns bytes and filename."""
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
    zip_bytes = buf.getvalue()

    # Automatically save a local file in backups/ folder
    try:
        save_backup_locally(filename, zip_bytes)
    except Exception as e:
        print(f"Warning: Could not save local backup file: {e}")

    return zip_bytes, filename


def get_local_backups() -> list[dict]:
    """Returns a list of local backup files sorted by modification time, newest first."""
    b_dir = get_backups_dir()
    if not os.path.exists(b_dir):
        return []
    backups = []
    try:
        for fname in os.listdir(b_dir):
            if fname.lower().endswith(".zip"):
                full_path = os.path.join(b_dir, fname)
                if os.path.isfile(full_path):
                    mtime = os.path.getmtime(full_path)
                    size_bytes = os.path.getsize(full_path)
                    dt = datetime.fromtimestamp(mtime)
                    date_str = dt.strftime("%d.%m.%Y %H:%M")
                    size_kb = max(1, round(size_bytes / 1024))
                    backups.append({
                        "filename": fname,
                        "path": full_path,
                        "mtime": mtime,
                        "date_str": date_str,
                        "size_str": f"{size_kb} KB",
                    })
    except Exception as e:
        print(f"Error reading backups directory: {e}")

    backups.sort(key=lambda b: b["mtime"], reverse=True)
    return backups


def delete_local_backup(filename: str) -> tuple[bool, str]:
    """Deletes a local backup file from the backups/ folder."""
    try:
        b_dir = get_backups_dir()
        clean_name = os.path.basename(filename)
        target_path = os.path.join(b_dir, clean_name)
        if os.path.exists(target_path):
            os.remove(target_path)
            return True, ""
        return False, "Datei nicht gefunden."
    except Exception as e:
        return False, str(e)


def restore_backup_zip(backup_source) -> tuple[bool, str]:
    """Restores presets and static settings from a zip file path or UploadedFile/BytesIO."""
    try:
        with zipfile.ZipFile(backup_source, "r") as zf:
            file_names = zf.namelist()
            presets_member = next((m for m in file_names if os.path.basename(m) == PRESETS_FILE), None)
            static_member = next((m for m in file_names if os.path.basename(m) == STATIC_CONFIG_FILE), None)

            if not presets_member and not static_member:
                return False, f"Weder '{PRESETS_FILE}' noch '{STATIC_CONFIG_FILE}' in der ZIP-Datei gefunden."

            if presets_member:
                presets_raw = zf.read(presets_member).decode("utf-8")
                presets_json = json.loads(presets_raw)
                presets_json = migrate_presets(presets_json)
                atomic_write_json(PRESETS_FILE, presets_json)

            if static_member:
                static_raw = zf.read(static_member).decode("utf-8")
                static_json = json.loads(static_raw)
                write_static_config_to_disk(static_json)

        return True, ""
    except Exception as e:
        return False, str(e)
