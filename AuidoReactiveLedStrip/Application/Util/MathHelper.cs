using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Util
{
    public static class MathHelper
    {
        private static uint rngState = 123456789;

        public static float Clamp(float value, float min, float max)
        {
            if (value < min)
            {
                return min;
            } 
            else if (value > max)
            {
                return max;
            }

            return value;
        }

        public static float Lerp(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        public static float NextFloatSigned()
        {
            return (NextUInt() / (float)uint.MaxValue) * 2f - 1f;
        }

        private static uint NextUInt()
        {
            uint x = rngState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rngState = x;
            return x;
        }

        public static float PyMod(float x, float m)
        {
            return (x % m + m) % m;
        }
    }
}
