using Application.Domain;
using Application.RuntimeSettings;
using Microsoft.Extensions.Options;
using System.Drawing;

namespace Application.Coloring.Remapping.Service
{
    public class RemapService : IRemapService
    {
        private readonly IOptionsMonitor<DynamicSettings> dynamicSettings;
        private readonly Accelerator accelerator;
        private readonly Patternizer patternizer;
        private readonly Repeater repeater;

        public RemapService(
            IOptionsMonitor<DynamicSettings> dynamicSettings,
            Accelerator accelerator,
            Patternizer patternizer,
            Repeater repeater
        )
        {
            this.dynamicSettings = dynamicSettings;
            this.accelerator = accelerator;
            this.patternizer = patternizer;
            this.repeater = repeater;
        }

        public Color[] RemapColors(LedStrip ledStrip)
        {
            DynamicSettings dynamicSettings = this.dynamicSettings.CurrentValue;

            Color[] colors = ledStrip.LedPixels.Select(l => l.Color).ToArray();
            colors = this.accelerator.Remap(colors, dynamicSettings);
            colors = this.patternizer.Remap(colors, dynamicSettings);
            return this.repeater.Remap(colors, dynamicSettings);
        }
    }
}
