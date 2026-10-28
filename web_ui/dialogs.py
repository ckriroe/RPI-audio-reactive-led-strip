import uuid
import streamlit as st
from web_ui.storage import write_presets_to_disk
from web_ui.preset_service import (
    get_group_name,
    get_preset_name,
    generate_copy_name,
    delete_group_by_id,
    delete_preset_by_id,
    copy_preset_by_id,
)
from web_ui.system_service import is_raspberry_pi, shutdown_system
from web_ui.backup_service import create_backup_zip, restore_backup_zip
from web_ui.storage import load_presets, load_static_settings
from web_ui.state_sync import (
    update_session_state_from_preset,
    update_session_state_from_static_settings,
)


def on_dialog_dismiss():
    st.session_state.active_dialog = None
    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
    st.session_state.pop("dlg_create_grp_name", None)
    st.session_state.pop("backup_data", None)
    st.session_state.pop("backup_filename", None)
    st.session_state.pop("import_backup_error", None)


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


@st.dialog("System herunterfahren", on_dismiss=on_dialog_dismiss)
def dialog_shutdown():
    st.write("Möchtest du das System wirklich herunterfahren?")
    on_rpi = is_raspberry_pi()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_shutdown_cancel", use_container_width=True):
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.button("Ja, herunterfahren", key="dlg_shutdown_submit", type="primary", use_container_width=True):
            if on_rpi:
                success, msg = shutdown_system()
                if success:
                    st.session_state.is_shutting_down = True
                    st.session_state.active_dialog = None
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.session_state.active_dialog = None
                st.rerun()


@st.dialog("Backup erstellen", on_dismiss=on_dialog_dismiss)
def dialog_create_backup():
    st.write("Erstelle ein Backup der aktuellen Einstellungen.")

    if "backup_data" not in st.session_state or st.session_state.get("backup_dialog_ver") != st.session_state.get("popover_ver"):
        zip_bytes, filename = create_backup_zip()
        st.session_state.backup_data = zip_bytes
        st.session_state.backup_filename = filename
        st.session_state.backup_dialog_ver = st.session_state.get("popover_ver")
    else:
        zip_bytes = st.session_state.backup_data
        filename = st.session_state.backup_filename

    st.info(f"Dateiname: **{filename}**")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Schließen", key="dlg_create_backup_close", use_container_width=True):
            st.session_state.pop("backup_data", None)
            st.session_state.pop("backup_filename", None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        if st.download_button(
            label="💾 Herunterladen",
            data=zip_bytes,
            file_name=filename,
            mime="application/zip",
            key="dlg_create_backup_download",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.toast_message = ("Backup erfolgreich erstellt!", "💾")
            st.session_state.pop("backup_data", None)
            st.session_state.pop("backup_filename", None)
            st.session_state.active_dialog = None
            st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
            st.rerun()


@st.dialog("Backup importieren", on_dismiss=on_dialog_dismiss)
def dialog_import_backup():
    st.write("Wähle eine Backup-ZIP-Datei aus, um die Einstellungen wiederherzustellen.")
    st.warning("⚠️ Die aktuellen Einstellungen werden vollständig überschrieben!")

    uploaded_file = st.file_uploader(
        "ZIP-Datei auswählen",
        type=["zip"],
        accept_multiple_files=False,
        key="dlg_import_backup_uploader",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Abbrechen", key="dlg_import_backup_cancel", use_container_width=True):
            st.session_state.pop("import_backup_error", None)
            st.session_state.active_dialog = None
            st.rerun()
    with col2:
        is_disabled = (uploaded_file is None)
        if st.button("Wiederherstellen", key="dlg_import_backup_submit", type="primary", disabled=is_disabled, use_container_width=True):
            success, err_msg = restore_backup_zip(uploaded_file)
            if success:
                st.session_state.pop("import_backup_error", None)
                st.session_state.active_dialog = None
                st.session_state.toast_message = ("Backup erfolgreich wiederhergestellt!", "✅")

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
                update_session_state_from_preset(st.session_state.presets[preset_index])

                static_data = load_static_settings()
                st.session_state.static_settings = static_data
                update_session_state_from_static_settings(static_data)

                st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
                st.rerun()
            else:
                st.session_state.import_backup_error = f"Fehler beim Wiederherstellen: {err_msg}"

    if st.session_state.get("import_backup_error"):
        st.error(st.session_state.import_backup_error)


def trigger_dialog(dlg_type, item_id=None):
    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
    st.session_state.active_dialog = {"type": dlg_type, "id": item_id}


def render_active_dialog():
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
        elif dtype == "shutdown":
            dialog_shutdown()
        elif dtype == "create_backup":
            dialog_create_backup()
        elif dtype == "import_backup":
            dialog_import_backup()
