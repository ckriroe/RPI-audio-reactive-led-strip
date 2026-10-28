from web_ui.constants import MIN_VOLUME_DB, MAX_VOLUME_DB


def migrate_groups(jsn: dict) -> None:
    if "groups" not in jsn:
        jsn["groups"] = []


def migrate_preset_group_ids(jsn: dict) -> None:
    presets = jsn.get("presets", [])
    for preset in presets:
        if "groupId" not in preset:
            preset["groupId"] = None

    if presets and len(presets) > 0:
        presets[0]["groupId"] = None


def migrate_frequency_amplitudes(jsn: dict) -> None:
    for preset in jsn.get("presets", []):
        vals = preset.get("values", {})
        if "minFreqAmplitude" in vals and (
            vals["minFreqAmplitude"] < MIN_VOLUME_DB or vals["minFreqAmplitude"] > MAX_VOLUME_DB
        ):
            vals["minFreqAmplitude"] = MIN_VOLUME_DB
        if "maxFreqAmplitude" in vals and (
            vals["maxFreqAmplitude"] < MIN_VOLUME_DB or vals["maxFreqAmplitude"] > MAX_VOLUME_DB
        ):
            vals["maxFreqAmplitude"] = MAX_VOLUME_DB


def migrate_presets(jsn: dict) -> dict:
    """Migrates preset configuration data to the current schema format."""
    if not isinstance(jsn, dict):
        return jsn

    migrate_groups(jsn)
    migrate_preset_group_ids(jsn)
    migrate_frequency_amplitudes(jsn)

    return jsn
