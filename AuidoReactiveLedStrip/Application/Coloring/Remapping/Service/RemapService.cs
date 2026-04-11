using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Remapping.Service
{
    public class RemapService : IRemapService
    {
        private readonly Accelerator accelerator;
        private readonly Patternizer patternizer;
        private readonly Repeater repeater;

        private DynamicEffectSettings? dynamicSettings = null;
        private StaticSettings? staticSettings = null;

        public RemapService(
            Accelerator accelerator,
            Patternizer patternizer,
            Repeater repeater
        )
        {
            this.accelerator = accelerator;
            this.patternizer = patternizer;
            this.repeater = repeater;
        }

        public Color[]? RemapColors(LedStrip ledStrip)
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            Color[] colors = ledStrip.LedPixels.Select(l => l.Color).ToArray();
            colors = this.accelerator.Remap(colors, dynamicSettings, staticSettings);
            colors = this.patternizer.Remap(colors, dynamicSettings, staticSettings);
            return this.repeater.Remap(colors, dynamicSettings, staticSettings);
        }

        public void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;
        }
    }
}
