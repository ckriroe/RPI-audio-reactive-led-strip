using Application.Coloring.Noise;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class IndexColorMode : BaseColorMode
    {
        public IndexColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicEffectSettings dynamicSettings, float audioValue, int index, int length)
        {
            float value = index / (float)(length - 1);
            return base.NonAudioValueToColor(value, audioValue, index, length, dynamicSettings);
        }
    }
}
