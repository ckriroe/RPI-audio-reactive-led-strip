import os
import sys
import subprocess


def is_raspberry_pi() -> bool:
    """Checks whether the system is running on a Raspberry Pi / Raspberry Pi OS."""
    if sys.platform.startswith("linux"):
        return True

    return False


def shutdown_system() -> tuple[bool, str]:
    """Initiates a system shutdown if running on a Raspberry Pi."""
    if not is_raspberry_pi():
        return False, "Herunterfahren wird nur auf dem Raspberry Pi (Raspberry Pi OS) unterstützt."

    commands = [
        ["sudo", "shutdown", "-h", "now"],
        ["shutdown", "-h", "now"],
        ["sudo", "systemctl", "poweroff"],
        ["systemctl", "poweroff"],
        ["sudo", "poweroff"],
        ["poweroff"],
    ]

    last_error = ""
    for cmd in commands:
        try:
            subprocess.Popen(cmd)
            return True, "System wird heruntergefahren..."
        except FileNotFoundError:
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return False, f"Fehler beim Herunterfahren: {last_error or 'Kein passender Befehl gefunden.'}"
