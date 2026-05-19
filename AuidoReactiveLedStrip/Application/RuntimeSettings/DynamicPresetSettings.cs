using Microsoft.Extensions.Configuration;

namespace Application.RuntimeSettings
{
    public class DynamicPresetSettings
    {
        public required Guid SelectedPresetId { get; set; }

        public List<Preset> Presets { get; set; } = [];
    }

    public class Preset
    {
        public Guid Id { get; set; }

        public required string Name { get; set; }

        [ConfigurationKeyName("values")]
        public required TemplateEffectSettings TemplateEffectSettings { get; set; }

        public required DynamicEffectSettings EffectSettings { get; set; }
    }
}
