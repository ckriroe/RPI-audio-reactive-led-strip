using Microsoft.Extensions.Configuration;

namespace Application.RuntimeSettings
{
    public class DynamicPresetSettings
    {
        [ConfigurationKeyName("selected_index")]
        public required int SelectedPresetIndex { get; set; }

        public List<Preset> Presets { get; set; } = [];
    }

    public class Preset
    {
        public required string Name { get; set; }

        [ConfigurationKeyName("values")]
        public required DynamicEffectSettings EffectSettings { get; set; }
    }
}
