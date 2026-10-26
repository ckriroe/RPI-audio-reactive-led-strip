using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public abstract class BaseColorCycleColorMode : BaseColorMode
    {
        protected readonly ColorSetting[] currentPalette =
        [
            new ColorSetting { Color = string.Empty, Threshold = 0.0f },
            new ColorSetting { Color = string.Empty, Threshold = 1.0f }
        ];

        protected float phase = 0.0f;

        protected BaseColorCycleColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
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

        protected void UpdatePaletteForPhase(DynamicEffectSettings dynamicSettings, float currentPhase)
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

            int secondaryCount = colors.Count - 1;
            if (secondaryCount == 1)
            {
                this.currentPalette[1].ColorInstance = colors[1].ColorInstance;
                this.currentPalette[1].Threshold = 1.0f;
                return;
            }

            float totalWeight = 0.0f;
            for (int i = 1; i < colors.Count; i++)
            {
                totalWeight += Math.Max(0.001f, colors[i].Threshold);
            }

            currentPhase = MathHelper.PyMod(currentPhase, 1.0f);

            float currentStart = 0.0f;
            int activeIdx = 0;
            float activeStart = 0.0f;
            float activeEnd = 1.0f;

            for (int i = 0; i < secondaryCount; i++)
            {
                float segLength = Math.Max(0.001f, colors[i + 1].Threshold) / totalWeight;
                float segEnd = currentStart + segLength;
                if (i == secondaryCount - 1)
                    segEnd = 1.0f;

                if (currentPhase >= currentStart && (currentPhase < segEnd || i == secondaryCount - 1))
                {
                    activeIdx = i;
                    activeStart = currentStart;
                    activeEnd = segEnd;
                    break;
                }

                currentStart = segEnd;
            }

            Color activeColor = colors[activeIdx + 1].ColorInstance;
            float transition = dynamicSettings.ColorTransition;

            if (transition > 0.0f)
            {
                float activeLength = activeEnd - activeStart;
                int nextIdx = (activeIdx + 1) % secondaryCount;
                int prevIdx = (activeIdx - 1 + secondaryCount) % secondaryCount;

                float nextLength = Math.Max(0.001f, colors[nextIdx + 1].Threshold) / totalWeight;
                float prevLength = Math.Max(0.001f, colors[prevIdx + 1].Threshold) / totalWeight;

                float tNext = Math.Min(activeLength, nextLength) * transition * 0.5f;
                float tPrev = Math.Min(activeLength, prevLength) * transition * 0.5f;

                if (tNext > 0.0f && currentPhase >= activeEnd - tNext)
                {
                    float blendProgress = (currentPhase - (activeEnd - tNext)) / (2.0f * tNext);
                    activeColor = ColorHelper.LerpColor(activeColor, colors[nextIdx + 1].ColorInstance, blendProgress);
                }
                else if (tPrev > 0.0f && currentPhase < activeStart + tPrev)
                {
                    float blendProgress = 0.5f + (currentPhase - activeStart) / (2.0f * tPrev);
                    activeColor = ColorHelper.LerpColor(colors[prevIdx + 1].ColorInstance, activeColor, blendProgress);
                }
            }

            this.currentPalette[1].ColorInstance = activeColor;
            this.currentPalette[1].Threshold = 1.0f;
        }
    }
}
