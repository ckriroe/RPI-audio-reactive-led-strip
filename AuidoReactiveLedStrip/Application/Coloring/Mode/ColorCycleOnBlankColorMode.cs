using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class ColorCycleOnBlankColorMode : BaseColorMode
    {
        private readonly ColorSetting[] currentPalette =
        [
            new ColorSetting { Color = string.Empty, Threshold = 0.0f },
            new ColorSetting { Color = string.Empty, Threshold = 1.0f }
        ];

        private int secondaryIndex = 1;
        private bool wasBlank = false;

        public ColorCycleOnBlankColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicEffectSettings dynamicSettings, float audioValue, int index, int length)
        {
            return ColorHelper.ValueToColor(
                LedHelper.ApplyNoise(audioValue, base.noiseGenerator.GetSmoothNoise(index, length, dynamicSettings)),
                this.currentPalette,
                dynamicSettings
            );
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings, LedStrip ledStrip)
        {
            var colors = dynamicSettings.Colors;
            if (colors.Count == 0)
            {
                this.currentPalette[0].ColorInstance = Color.Black;
                this.currentPalette[1].ColorInstance = Color.Black;
                return;
            }

            var bgEntry = colors[0];
            this.currentPalette[0].ColorInstance = bgEntry.ColorInstance;
            this.currentPalette[0].Threshold = bgEntry.Threshold;

            if (colors.Count == 1)
            {
                this.currentPalette[1].ColorInstance = bgEntry.ColorInstance;
                this.currentPalette[1].Threshold = 1.0f;
                return;
            }

            bool isBlank = true;
            float minVal = staticSettings.MinSanitizedValue;
            var pixels = ledStrip.LedPixels;
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixels[i].Value > minVal)
                {
                    isBlank = false;
                    break;
                }
            }

            if (isBlank && !this.wasBlank)
            {
                this.secondaryIndex++;
                if (this.secondaryIndex >= colors.Count)
                    this.secondaryIndex = 1;
            }

            if (this.secondaryIndex >= colors.Count || this.secondaryIndex < 1)
                this.secondaryIndex = 1;

            this.currentPalette[1].ColorInstance = colors[this.secondaryIndex].ColorInstance;
            this.currentPalette[1].Threshold = 1.0f;

            this.wasBlank = isBlank;
        }
    }
}
