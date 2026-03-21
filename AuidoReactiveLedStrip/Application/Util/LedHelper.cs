using Application.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Util
{
    public static class LedHelper
    {
        public static LedStrip CreateFilledStrip(int length, float value)
        {
            return new LedStrip()
            {
                LedPixels = [.. Enumerable.Range(0, length).Select(_ => new LedPixel(value))]
            };
        }

        public static LedStrip CreateEmptyStrip(int lenfth)
        {
            return CreateFilledStrip(lenfth, 0.0f);
        }

        public static float ApplyNoise(float value, float noise)
        {
            return MathHelper.Clamp(value + noise, 0.0f, 1.0f);
        }
    }
}
