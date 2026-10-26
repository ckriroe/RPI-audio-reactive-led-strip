using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Coloring.Mode
{
    public class ColorCycleByTimeColorMode : BaseColorCycleColorMode
    {
        public ColorCycleByTimeColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings, LedStrip ledStrip)
        {
            float speed = dynamicSettings.ColorCycleSpeed;
            float fps = Math.Max(1, staticSettings.Fps);
            base.phase = MathHelper.PyMod(base.phase + (speed / fps), 1.0f);
            base.UpdatePaletteForPhase(dynamicSettings, base.phase);
        }
    }
}
