import os
import sys
import json
import zipfile
import tempfile
import subprocess
import urllib.request
import urllib.error
from web_ui.storage import load_version
from web_ui.system_service import is_raspberry_pi

GITHUB_REPO = "ckriroe/RPI-audio-reactive-led-strip"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
DEFAULT_DOWNLOAD_URL = f"https://github.com/{GITHUB_REPO}/releases/latest/download/app.zip"


def parse_version(ver_str: str) -> tuple[int, ...]:
    """Parses a version string into an integer tuple for reliable comparison."""
    if not ver_str:
        return (0, 0, 0)
    cleaned = ver_str.strip().lstrip("vV")
    parts = []
    for part in cleaned.split("."):
        num_str = "".join(ch for ch in part if ch.isdigit())
        if num_str:
            parts.append(int(num_str))
        else:
            parts.append(0)
    return tuple(parts) if parts else (0, 0, 0)


def check_for_updates() -> dict:
    """Queries GitHub for the latest release and compares with the currently installed version."""
    cur_version = load_version() or "0.0.0"
    try:
        req = urllib.request.Request(
            LATEST_RELEASE_API,
            headers={
                "User-Agent": "RPISC-WebUI",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            if response.status != 200:
                return {
                    "success": False,
                    "error": f"GitHub API antwortete mit Statuscode {response.status}",
                    "current_version": cur_version,
                }
            data = json.loads(response.read().decode("utf-8"))

        raw_tag = data.get("tag_name", "")
        latest_version = raw_tag.lstrip("vV")
        body_text = data.get("body", "")

        download_url = DEFAULT_DOWNLOAD_URL
        for asset in data.get("assets", []):
            if asset.get("name") == "app.zip":
                download_url = asset.get("browser_download_url", DEFAULT_DOWNLOAD_URL)
                break

        current_parsed = parse_version(cur_version)
        latest_parsed = parse_version(latest_version)

        has_update = latest_parsed > current_parsed

        return {
            "success": True,
            "current_version": cur_version,
            "latest_version": latest_version,
            "has_update": has_update,
            "download_url": download_url,
            "release_notes": body_text,
        }
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return {
                "success": False,
                "error": "GitHub API Rate-Limit erreicht. Bitte später erneut versuchen.",
                "current_version": cur_version,
            }
        return {
            "success": False,
            "error": f"HTTP-Fehler {e.code}: {e.reason}",
            "current_version": cur_version,
        }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": f"Verbindungsfehler: Keine Internetverbindung oder GitHub nicht erreichbar.",
            "current_version": cur_version,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unerwarteter Fehler: {str(e)}",
            "current_version": cur_version,
        }


def get_update_staging_dir() -> str:
    """Returns the temporary staging directory for updates."""
    if is_raspberry_pi():
        return "/tmp/audio-system-update"
    return os.path.join(tempfile.gettempdir(), "audio-system-update")


def download_update_package(download_url: str = None) -> tuple[bool, str]:
    """Downloads the release app.zip and verifies its integrity."""
    target_dir = get_update_staging_dir()
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "app.zip")

    url = download_url or DEFAULT_DOWNLOAD_URL

    try:
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
            except Exception:
                pass

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "RPISC-WebUI",
                "Accept": "application/octet-stream",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(target_file, "wb") as out_file:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)

        # Validate archive integrity
        if not zipfile.is_zipfile(target_file):
            return False, "Die heruntergeladene Datei ist kein gültiges ZIP-Archiv."

        with zipfile.ZipFile(target_file, "r") as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                return False, f"ZIP-Archiv ist beschädigt (Fehler in {corrupt})."

        return True, ""
    except urllib.error.URLError as e:
        return False, f"Netzwerkfehler beim Herunterladen: {str(e)}"
    except Exception as e:
        return False, f"Download-Fehler: {str(e)}"


def trigger_system_update() -> tuple[bool, str]:
    """Triggers the system update service via systemd."""
    if not is_raspberry_pi():
        return True, "Simulation: Update auf Desktop erfolgreich simuliert."

    commands = [
        ["systemctl", "start", "--no-block", "audio-update.service"],
        ["sudo", "systemctl", "start", "--no-block", "audio-update.service"],
    ]

    last_error = ""
    for cmd in commands:
        try:
            subprocess.Popen(cmd)
            return True, "Update-Dienst gestartet."
        except FileNotFoundError:
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return False, f"Fehler beim Starten des Update-Dienstes: {last_error or 'Befehl systemctl nicht gefunden.'}"
