using Application.Coloring.Noise;
using Application.Settings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class IndexColorMode : BaseColorMode
    {
        public IndexColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicSettings dynamicSettings, float audioValue, int index, int length)
        {
            float value = index / (float)(length - 1);
            return base.NonAudioValueToColor(value, audioValue, index, length, dynamicSettings);
        }
    }
}
