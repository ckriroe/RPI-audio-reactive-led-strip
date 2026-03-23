using Application.Coloring.Noise;
using Application.Domain;
using Application.Settings;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class DistanceToCenterColorMode : BaseColorMode
    {
        
        private float maxDistance;

        public DistanceToCenterColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicSettings dynamicSettings, float audioValue, int index, int length)
        {
            int distance = Math.Abs(index - dynamicSettings.EffectOrigin);
            float value = distance / this.maxDistance;
            return base.NonAudioValueToColor(value, audioValue, index, length, dynamicSettings);
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicSettings dynamicSettings, LedStrip ledStrip)
        {
            this.maxDistance = Math.Max(dynamicSettings.EffectOrigin, (ledStrip.LedPixels.Length - 1) - dynamicSettings.EffectOrigin);
            if (maxDistance <= 0.0f)
                this.maxDistance = 1.0f;
        }
    }
}
