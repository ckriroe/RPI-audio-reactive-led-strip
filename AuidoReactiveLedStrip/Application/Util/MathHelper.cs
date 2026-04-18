using Application.Domain;

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

        public static float Lerp(float a, float b, float t, EasingFunctionType easing = EasingFunctionType.Linear)
        {
            return a + (b - a) * Easing.ApplyEasing(easing, t);
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

        public static class Easing
        {
            public static float ApplyEasing(EasingFunctionType type, float t)
            {
                return type switch
                {
                    EasingFunctionType.Linear => t,

                    EasingFunctionType.EaseInQuad => t * t,
                    EasingFunctionType.EaseOutQuad => 1 - (1 - t) * (1 - t),
                    EasingFunctionType.EaseInOutQuad =>
                        t < 0.5f ? 2 * t * t : 1 - MathF.Pow(-2 * t + 2, 2) / 2,

                    EasingFunctionType.EaseInCubic => t * t * t,
                    EasingFunctionType.EaseOutCubic => 1 - MathF.Pow(1 - t, 3),
                    EasingFunctionType.EaseInOutCubic =>
                        t < 0.5f ? 4 * t * t * t : 1 - MathF.Pow(-2 * t + 2, 3) / 2,

                    EasingFunctionType.EaseInQuart => t * t * t * t,
                    EasingFunctionType.EaseOutQuart => 1 - MathF.Pow(1 - t, 4),
                    EasingFunctionType.EaseInOutQuart =>
                        t < 0.5f ? 8 * t * t * t * t : 1 - MathF.Pow(-2 * t + 2, 4) / 2,

                    EasingFunctionType.EaseInQuint => t * t * t * t * t,
                    EasingFunctionType.EaseOutQuint => 1 - MathF.Pow(1 - t, 5),
                    EasingFunctionType.EaseInOutQuint =>
                        t < 0.5f ? 16 * t * t * t * t * t : 1 - MathF.Pow(-2 * t + 2, 5) / 2,

                    EasingFunctionType.EaseInSine => 1 - MathF.Cos((t * MathF.PI) / 2),
                    EasingFunctionType.EaseOutSine => MathF.Sin((t * MathF.PI) / 2),
                    EasingFunctionType.EaseInOutSine =>
                        -(MathF.Cos(MathF.PI * t) - 1) / 2,

                    EasingFunctionType.EaseInExpo =>
                        t == 0 ? 0 : MathF.Pow(2, 10 * t - 10),
                    EasingFunctionType.EaseOutExpo =>
                        t == 1 ? 1 : 1 - MathF.Pow(2, -10 * t),
                    EasingFunctionType.EaseInOutExpo =>
                        t == 0 ? 0 :
                        t == 1 ? 1 :
                        t < 0.5f
                            ? MathF.Pow(2, 20 * t - 10) / 2
                            : (2 - MathF.Pow(2, -20 * t + 10)) / 2,

                    EasingFunctionType.EaseInCirc =>
                        1 - MathF.Sqrt(1 - t * t),
                    EasingFunctionType.EaseOutCirc =>
                        MathF.Sqrt(1 - MathF.Pow(t - 1, 2)),
                    EasingFunctionType.EaseInOutCirc =>
                        t < 0.5f
                            ? (1 - MathF.Sqrt(1 - MathF.Pow(2 * t, 2))) / 2
                            : (MathF.Sqrt(1 - MathF.Pow(-2 * t + 2, 2)) + 1) / 2,

                    EasingFunctionType.EaseOutBounce => EaseOutBounce(t),
                    EasingFunctionType.EaseInBounce => 1f - EaseOutBounce(1f - t),
                    EasingFunctionType.EaseInOutBounce =>
                        t < 0.5f
                            ? (1f - EaseOutBounce(1f - 2f * t)) / 2f
                            : (1f + EaseOutBounce(2f * t - 1f)) / 2f,

                    EasingFunctionType.EaseOutElastic => EaseOutElastic(t),
                    EasingFunctionType.EaseInElastic => EaseInElastic(t),
                    EasingFunctionType.EaseInOutElastic => EaseInOutElastic(t),

                    EasingFunctionType.EaseOutBack => EaseOutBack(t),
                    EasingFunctionType.EaseInBack => EaseInBack(t),
                    EasingFunctionType.EaseInOutBack => EaseInOutBack(t),

                    EasingFunctionType.SmoothStep =>
                        t * t * (3 - 2 * t),

                    EasingFunctionType.SmootherStep =>
                        t * t * t * (t * (6 * t - 15) + 10),

                    EasingFunctionType.EaseInOutSigmoid =>
                        Sigmoid(t),

                    _ => t
                };
            }

            private static float EaseOutBounce(float t)
            {
                const float n1 = 7.5625f;
                const float d1 = 2.75f;

                if (t < 1 / d1)
                    return n1 * t * t;
                else if (t < 2 / d1)
                    return n1 * (t -= 1.5f / d1) * t + 0.75f;
                else if (t < 2.5f / d1)
                    return n1 * (t -= 2.25f / d1) * t + 0.9375f;
                else
                    return n1 * (t -= 2.625f / d1) * t + 0.984375f;
            }

            private static float EaseOutElastic(float t)
            {
                const float c4 = (2 * MathF.PI) / 3;

                if (t == 0) return 0;
                if (t == 1) return 1;

                return MathF.Pow(2, -10 * t) * MathF.Sin((t * 10 - 0.75f) * c4) + 1;
            }

            private static float EaseInElastic(float t)
            {
                const float c4 = (2 * MathF.PI) / 3;

                if (t == 0) return 0;
                if (t == 1) return 1;

                return -MathF.Pow(2, 10 * t - 10) * MathF.Sin((t * 10 - 10.75f) * c4);
            }

            private static float EaseInOutElastic(float t)
            {
                const float c5 = (2 * MathF.PI) / 4.5f;

                if (t == 0) return 0;
                if (t == 1) return 1;

                return t < 0.5f
                    ? -(MathF.Pow(2, 20 * t - 10) * MathF.Sin((20 * t - 11.125f) * c5)) / 2
                    : (MathF.Pow(2, -20 * t + 10) * MathF.Sin((20 * t - 11.125f) * c5)) / 2 + 1;
            }

            private static float EaseOutBack(float t)
            {
                const float c1 = 1.70158f;
                const float c3 = c1 + 1;

                return 1 + c3 * MathF.Pow(t - 1, 3) + c1 * MathF.Pow(t - 1, 2);
            }

            private static float EaseInBack(float t)
            {
                const float c1 = 1.70158f;
                const float c3 = c1 + 1;

                return c3 * t * t * t - c1 * t * t;
            }

            private static float EaseInOutBack(float t)
            {
                const float c1 = 1.70158f;
                const float c2 = c1 * 1.525f;

                return t < 0.5f
                    ? (MathF.Pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
                    : (MathF.Pow(2 * t - 2, 2) * ((c2 + 1) * (2 * t - 2) + c2) + 2) / 2;
            }

            private static float Sigmoid(float t)
            {
                float k = 10f;
                float x = t * 2 - 1;
                float y = 1f / (1f + MathF.Exp(-k * x));
                float y0 = 1f / (1f + MathF.Exp(k));
                float y1 = 1f / (1f + MathF.Exp(-k));
                return (y - y0) / (y1 - y0);
            }
        }
    }
}
