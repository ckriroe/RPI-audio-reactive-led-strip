using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Util
{
    public static class MathHelper
    {
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
    }
}
