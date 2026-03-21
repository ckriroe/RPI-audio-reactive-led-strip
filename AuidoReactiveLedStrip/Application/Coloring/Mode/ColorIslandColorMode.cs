using Application.Coloring.Noise;
using Application.Domain;
using Application.Settings;
using Application.Util;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public class ColorIslandColorMode : BaseColorMode
    {
        float[] preComutedValues = Array.Empty<float>();

        public ColorIslandColorMode(INoiseGenerator noiseGenerator) : base(noiseGenerator)
        {
        }

        public override Color GetColorForValue(DynamicSettings dynamicSettings, float audioValue, int index, int length)
        {
            return base.NonAudioValueToColor(
                this.preComutedValues[index],
                audioValue,
                index,
                length,
                dynamicSettings
            );
        }

        public override void PrecomputeValues(StaticSettings staticSettings, DynamicSettings dynamicSettings, LedStrip ledStrip)
        {
            int n = ledStrip.LedPixels.Count;
            if (this.preComutedValues.Length != n)
                this.preComutedValues = new float[n];

            var preComutedValues = this.preComutedValues;
            int i = 0;
            while (i < n)
            {
                LedPixel pixelAtIndex = ledStrip.LedPixels[i];
                if (pixelAtIndex.Value <= 0.0)
                {
                    preComutedValues[i] = 0.0f;
                    i += 1;
                    continue;
                }

                int start = i;
                while (i < n && pixelAtIndex.Value > 0.0)
                {
                    i += 1;
                }

                int end = i - 1;
                int length = end - start + 1;
                if (length == 1)
                {
                    preComutedValues[start] = 1.0f;
                    continue;
                }

                float mid = (length - 1) / 2.0f;
                for (int j = 0; j < length; j++)
                {
                    int idx = start + j;
                    float dist = mid != 0.0f ? Math.Abs(j - mid) / mid : 0.0f;
                    float value = MathHelper.Clamp(1.0f - dist, 0.0f, 1.0f);
                    preComutedValues[idx] = value;
                }
            }
        }
    }
}
