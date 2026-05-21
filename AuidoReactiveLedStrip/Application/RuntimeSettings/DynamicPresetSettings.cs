using Microsoft.Extensions.Configuration;

namespace Application.RuntimeSettings
{
    public class DynamicPresetSettings
    {
        public required Guid SelectedPresetId { get; set; }

        public List<Preset> Presets { get; set; } = [];

        public override bool Equals(object? obj)
        {
            return obj is DynamicPresetSettings settings &&
                   this.SelectedPresetId.Equals(settings.SelectedPresetId) &&
                   this.Presets.SequenceEqual(settings.Presets);
        }

        public override int GetHashCode()
        {
            HashCode hash = new();

            hash.Add(this.SelectedPresetId);
            foreach (Preset preset in this.Presets)
            {
                hash.Add(preset);
            }

            return hash.ToHashCode();
        }
    }

    public class Preset
    {
        public Guid Id { get; set; }

        public required string Name { get; set; }

        [ConfigurationKeyName("values")]
        public required TemplateEffectSettings TemplateEffectSettings { get; set; }

        public required DynamicEffectSettings EffectSettings { get; set; }

        public override bool Equals(object? obj)
        {
            return obj is Preset preset &&
                   this.Id.Equals(preset.Id) &&
                   this.Name == preset.Name &&
                   EqualityComparer<DynamicEffectSettings>.Default.Equals(this.EffectSettings, preset.EffectSettings);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(this.Id, this.Name, this.EffectSettings);
        }
    }
}
