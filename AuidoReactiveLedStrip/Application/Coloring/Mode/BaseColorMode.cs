using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public abstract class BaseColorMode : IColorMode
    {
        protected INoiseGenerator noiseGenerator;

        protected BaseColorMode(INoiseGenerator noiseGenerator)
        {
            this.noiseGenerator = noiseGenerator;
        }

        public abstract Color GetColorForValue(DynamicEffectSettings dynamicSettings, float audioValue, int index, int length);

        public virtual void PrecomputeValues(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings, LedStrip ledStrip)
        {
            // do nothing by default
        }

        protected Color NonAudioValueToColor(float value, float audioValue, int index, int length, DynamicEffectSettings dynamicSettings)
        {
            float noise = this.noiseGenerator.GetSmoothNoise(index, length, dynamicSettings);
            return ColorHelper.NonAudioValueToColor(
                LedHelper.ApplyNoise(value, noise),
                LedHelper.ApplyNoise(audioValue, noise),
                dynamicSettings
            );
        }
    }
}
