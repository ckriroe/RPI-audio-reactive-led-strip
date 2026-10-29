import uuid
import base64
import streamlit as st
import streamlit.components.v1 as components
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
from web_ui.backup_service import (
    create_backup_zip,
    restore_backup_zip,
    get_local_backups,
    delete_local_backup,
)
from web_ui.storage import load_presets, load_static_settings
from web_ui.state_sync import (
    update_session_state_from_preset,
    update_session_state_from_static_settings,
)
from web_ui.update_service import (
    check_for_updates,
    download_update_package,
    trigger_system_update,
)


def on_dialog_dismiss():
    st.session_state.active_dialog = None
    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
    st.session_state.pop("dlg_create_grp_name", None)
    st.session_state.pop("backup_data", None)
    st.session_state.pop("backup_filename", None)
    st.session_state.pop("import_backup_error", None)
    st.session_state.pop("confirm_restore_backup", None)
    st.session_state.pop("confirm_delete_backup", None)
    st.session_state.pop("update_info", None)
    st.session_state.pop("update_step", None)
    st.session_state.pop("update_backup_data", None)
    st.session_state.pop("update_backup_filename", None)
    st.session_state.pop("update_backup_done", None)
    st.session_state.pop("update_error", None)
    st.session_state.pop("update_download_ok", None)


def apply_restored_settings():
    """Reloads presets and static configuration into session state after a backup restore."""
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

    st.info(f"Dateiname: **{filename}**  \n*(Automatisch im Ordner 'backups/' gespeichert)*")

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


@st.dialog("Backup wiederherstellen", on_dismiss=on_dialog_dismiss)
def dialog_import_backup():
    # 1. Confirmation View: Restore
    if st.session_state.get("confirm_restore_backup"):
        target = st.session_state.confirm_restore_backup
        st.warning("⚠️ **Wiederherstellung bestätigen**")
        st.write(f"Möchtest du das Backup **{target['name']}** wirklich wiederherstellen?")
        st.error("Alle aktuellen Einstellungen und Voreinstellungen werden dabei überschrieben!")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Abbrechen", key="btn_conf_rst_cancel", use_container_width=True):
                st.session_state.pop("confirm_restore_backup", None)
                st.rerun()
        with col2:
            if st.button("Ja, wiederherstellen", key="btn_conf_rst_submit", type="primary", use_container_width=True):
                success, err_msg = restore_backup_zip(target["source"])
                if success:
                    st.session_state.pop("confirm_restore_backup", None)
                    st.session_state.pop("import_backup_error", None)
                    st.session_state.active_dialog = None
                    st.session_state.toast_message = ("Backup erfolgreich wiederhergestellt!", "✅")
                    apply_restored_settings()
                    st.session_state.popover_ver = st.session_state.get("popover_ver", 0) + 1
                    st.rerun()
                else:
                    st.session_state.import_backup_error = f"Fehler beim Wiederherstellen: {err_msg}"
                    st.session_state.pop("confirm_restore_backup", None)
                    st.rerun()
        return

    # 2. Confirmation View: Delete
    if st.session_state.get("confirm_delete_backup"):
        del_name = st.session_state.confirm_delete_backup
        st.warning(f"Möchtest du das lokale Backup **{del_name}** wirklich dauerhaft löschen?")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Abbrechen", key="btn_conf_del_cancel", use_container_width=True):
                st.session_state.pop("confirm_delete_backup", None)
                st.rerun()
        with col2:
            if st.button("Ja, löschen", key="btn_conf_del_submit", type="primary", use_container_width=True):
                ok, err = delete_local_backup(del_name)
                st.session_state.pop("confirm_delete_backup", None)
                if ok:
                    st.session_state.toast_message = ("Backup gelöscht.", "🗑️")
                else:
                    st.session_state.import_backup_error = f"Fehler beim Löschen: {err}"
                st.rerun()
        return

    # 3. Main View with Tabs
    st.write("Wähle ein Backup aus, um deine Einstellungen wiederherzustellen.")

    tab_local, tab_upload = st.tabs(["📁 Lokale Backups", "📤 Datei hochladen"])

    with tab_local:
        local_backups = get_local_backups()
        if not local_backups:
            st.info("Noch keine lokalen Backups vorhanden.")
        else:
            st.caption(f"{len(local_backups)} Backup(s) gefunden (neueste zuerst):")
            with st.container(height=260):
                for b in local_backups:
                    fname = b["filename"]
                    col_info, col_rst, col_dl, col_del = st.columns([6, 2, 2, 2], vertical_alignment="center", gap="small")
                    with col_info:
                        st.markdown(f"**{b['date_str']}**  \n<small title='{fname}'>{b['size_str']}</small>", unsafe_allow_html=True)
                    with col_rst:
                        if st.button("🔄", key=f"btn_rst_{fname}"):
                            st.session_state.confirm_restore_backup = {
                                "name": fname,
                                "source": b["path"],
                            }
                            st.rerun()
                    with col_dl:
                        with open(b["path"], "rb") as f:
                            dl_bytes = f.read()
                        st.download_button(
                            "💾",
                            data=dl_bytes,
                            file_name=fname,
                            mime="application/zip",
                            key=f"btn_dl_{fname}",
                        )
                    with col_del:
                        if st.button("🗑️", key=f"btn_del_{fname}"):
                            st.session_state.confirm_delete_backup = fname
                            st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "ZIP-Datei auswählen",
            type=["zip"],
            accept_multiple_files=False,
            key="dlg_import_backup_uploader",
        )
        is_disabled = (uploaded_file is None)
        if st.button("Wiederherstellen", key="dlg_import_upload_submit", type="primary", disabled=is_disabled, use_container_width=True):
            st.session_state.confirm_restore_backup = {
                "name": uploaded_file.name,
                "source": uploaded_file,
            }
            st.rerun()

    if st.session_state.get("import_backup_error"):
        st.error(st.session_state.import_backup_error)

    if st.button("Schließen", key="dlg_import_backup_close", use_container_width=True):
        st.session_state.pop("import_backup_error", None)
        st.session_state.active_dialog = None
        st.rerun()


@st.dialog("Nach Updates suchen", on_dismiss=on_dialog_dismiss)
def dialog_check_update():
    step = st.session_state.get("update_step", "check")

    if step == "check":
        if "update_info" not in st.session_state:
            with st.spinner("Suche nach verfügbaren Updates..."):
                st.session_state.update_info = check_for_updates()

        info = st.session_state.update_info

        if not info.get("success"):
            st.error(f"Fehler bei der Update-Prüfung:\n{info.get('error')}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Schließen", key="dlg_upd_err_close", use_container_width=True):
                    st.session_state.active_dialog = None
                    st.rerun()
            with col2:
                if st.button("Erneut versuchen", key="dlg_upd_err_retry", type="primary", use_container_width=True):
                    st.session_state.pop("update_info", None)
                    st.rerun()
            return

        if not info.get("has_update"):
            st.success(f"Du verwendest bereits die neueste Version (**v{info.get('current_version')}**).")
            if st.button("Schließen", key="dlg_upd_noupd_close", use_container_width=True):
                st.session_state.active_dialog = None
                st.rerun()
            return

        # Update available
        st.info(f"Eine neue Version ist verfügbar: **v{info.get('latest_version')}**\n\nAktuell installierte Version: **v{info.get('current_version')}**")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Schließen", key="dlg_upd_avail_close", use_container_width=True):
                st.session_state.active_dialog = None
                st.rerun()
        with col2:
            if st.button("Installieren", key="dlg_upd_avail_install", type="primary", use_container_width=True):
                st.session_state.update_step = "confirm"
                st.rerun()

    elif step == "confirm":
        info = st.session_state.get("update_info", {})
        latest_ver = info.get("latest_version", "")
        st.subheader("Update bestätigen")
        st.write(f"Möchtest du das Update auf Version **v{latest_ver}** wirklich installieren?")
        st.write("Vor der Installation wird automatisch ein Backup erstellt.")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Abbrechen", key="dlg_upd_conf_cancel", use_container_width=True):
                st.session_state.active_dialog = None
                st.rerun()
        with col2:
            if st.button("Ja, jetzt aktualisieren", key="dlg_upd_conf_submit", type="primary", use_container_width=True):
                st.session_state.update_step = "installing"
                st.rerun()

    elif step == "installing":
        info = st.session_state.get("update_info", {})

        # 1. Create and auto-download backup if not already done
        if not st.session_state.get("update_backup_done"):
            zip_bytes, filename = create_backup_zip()
            st.session_state.update_backup_data = zip_bytes
            st.session_state.update_backup_filename = filename
            st.session_state.update_backup_done = True

            b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
            auto_dl_script = f"""
            <a id="auto_dl_link" href="data:application/zip;base64,{b64_zip}" download="{filename}" style="display:none;"></a>
            <script>
                var link = document.getElementById("auto_dl_link");
                if (link) {{
                    link.click();
                }}
            </script>
            """
            components.html(auto_dl_script, height=0)

        backup_file = st.session_state.get("update_backup_filename", "Backup")
        # 2. Download and verify update package if not already done
        if not st.session_state.get("update_download_ok") and not st.session_state.get("update_error"):
            with st.spinner("Lade Update-Paket herunter..."):
                dl_url = info.get("download_url")
                success, err_msg = download_update_package(dl_url)
                if success:
                    st.session_state.update_download_ok = True
                    st.session_state.pop("update_error", None)
                else:
                    st.session_state.update_error = err_msg

        # 3. If download failed, show error and retry button (backup is preserved!)
        if st.session_state.get("update_error"):
            st.error(f"Fehler beim Herunterladen des Updates:\n{st.session_state.update_error}")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Abbrechen", key="dlg_upd_inst_cancel", use_container_width=True):
                    st.session_state.active_dialog = None
                    st.rerun()
            with col2:
                if st.button("Erneut versuchen", key="dlg_upd_inst_retry", type="primary", use_container_width=True):
                    st.session_state.pop("update_error", None)
                    st.rerun()
            return

        # 4. Trigger system update service
        if st.session_state.get("update_download_ok"):
            st.info("Update-Paket verifiziert. Starte Update-Dienst...")
            svc_ok, svc_msg = trigger_system_update()
            if svc_ok:
                st.success("Das Update wird ausgeführt.")
                reload_script = """
                <script>
                    setTimeout(function() {
                        window.location.reload();
                    }, 15000);
                </script>
                """
                components.html(reload_script, height=0)
            else:
                st.error(f"Fehler beim Starten des Updates:\n{svc_msg}")
                if st.button("Schließen", key="dlg_upd_svc_err_close", use_container_width=True):
                    st.session_state.active_dialog = None
                    st.rerun()


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
        elif dtype == "check_update":
            dialog_check_update()
