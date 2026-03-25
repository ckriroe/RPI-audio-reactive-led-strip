using Application.Coloring.Noise;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class ColorWaveColorMode : BaseColorMode
    {
        private float colorWavePhase;

        public ColorWaveColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicSettings dynamicSettings, float audioValue, int index, int length)
        {
            float distance = Math.Abs(index - dynamicSettings.ColorWaveOrigin);
            float wavePosition = MathHelper.PyMod(distance + this.colorWavePhase, dynamicSettings.ColorWaveSize);
            float value = this.WaveEnvelope(wavePosition / dynamicSettings.ColorWaveSize);
            return base.NonAudioValueToColor(value, audioValue, index, length, dynamicSettings);
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicSettings dynamicSettings, LedStrip ledStrip)
        {
            int direction = dynamicSettings.ColorWaveInwards ? 1 : -1;
            this.colorWavePhase += direction * dynamicSettings.ColorWaveSpeed / (float)staticSettings.Fps;
            if (Math.Abs(colorWavePhase) >= dynamicSettings.ColorWaveSize)
                colorWavePhase -= dynamicSettings.ColorWaveSize * direction;
            //this.colorWavePhase = MathHelper.PyMod(this.colorWavePhase, dynamicSettings.ColorWaveSize);
        }

        private float WaveEnvelope(float x)
        {
            if (x < 0.5f)
                return x * 2.0f;

            return (1.0f - x) * 2.0f;
        }
    }
}
