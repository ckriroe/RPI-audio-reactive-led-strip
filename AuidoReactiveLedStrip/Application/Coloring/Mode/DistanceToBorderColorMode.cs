using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class DistanceToBorderColorMode : BaseColorMode
    {

        private float maxDistance;

        public DistanceToBorderColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicEffectSettings dynamicSettings, float audioValue, int index, int length)
        {
            int distance = Math.Min(index, (length - 1) - index);
            float value = distance / this.maxDistance;
            return base.NonAudioValueToColor(value, audioValue, index, length, dynamicSettings);
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings, LedStrip ledStrip)
        {
            this.maxDistance = (ledStrip.LedPixels.Length - 1) / 2.0f;
            if (maxDistance <= 0.0f)
                this.maxDistance = 1.0f;
        }
    }
}
