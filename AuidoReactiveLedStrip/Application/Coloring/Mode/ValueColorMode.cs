using Application.Coloring.Noise;
using Application.Domain;
using Application.Settings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class ValueColorMode : BaseColorMode
    {
        public ValueColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicSettings dynamicSettings, float pixelValue, int index, int length)
        {
            return ColorHelper.ValueToColor(
                LedHelper.ApplyNoise(pixelValue, base.noiseGenerator.GetSmoothNoise(index, length, dynamicSettings)),
                dynamicSettings
            );
        }
    }
}
