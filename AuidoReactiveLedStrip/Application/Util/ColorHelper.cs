using Application.Settings;
using MathNet.Numerics.Distributions;
using System.Drawing;

namespace Application.Util
{
    public static class ColorHelper
    {
        public static Color LerpColor(Color a, Color b, float t)
        {
            int r = Math.Max(0, (int)(a.R + (b.R - a.R) * t));
            int g = Math.Max(0, (int)(a.G + (b.G - a.G) * t));
            int bVal = Math.Max(0, (int)(a.B + (b.B - a.B) * t));

            return Color.FromArgb(r, g, bVal);
        }

        public static Color HsvToRgb(float h, float s, float v)
        {
            float r = 0, g = 0, b = 0;

            int i = (int)(h * 6);
            float f = h * 6 - i;
            float p = v * (1 - s);
            float q = v * (1 - f * s);
            float t = v * (1 - (1 - f) * s);

            switch (i % 6)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                case 5: r = v; g = p; b = q; break;
            }

            return Color.FromArgb(
                (int)(r * 255),
                (int)(g * 255),
                (int)(b * 255)
            );
        }

        public static void HsvToRgb(float h, float s, float v, out float r, out float g, out float b)
        {
            if (s == 0.0)
            {
                r = g = b = v;
                return;
            }

            float i = (float)Math.Floor(h * 6.0f);
            float f = h * 6.0f - i;
            float p = v * (1.0f - s);
            float q = v * (1.0f - f * s);
            float t = v * (1.0f - (1.0f - f) * s);

            switch ((int)i % 6)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                default: r = v; g = p; b = q; break;
            }
        }

        public static void RgbToHsv(float r, float g, float b, out float h, out float s, out float v)
        {
            float max = Math.Max(r, Math.Max(g, b));
            float min = Math.Min(r, Math.Min(g, b));
            v = max;

            float delta = max - min;

            if (max == 0.0f)
            {
                s = 0.0f;
                h = 0.0f;
                return;
            }

            s = delta / max;

            if (delta == 0.0f)
            {
                h = 0.0f;
            } else if (max == r)
            {
                h = (g - b) / delta % 6.0f;
            } else if (max == g)
            {
                h = (b - r) / delta + 2.0f;
            } else
            {
                h = (r - g) / delta + 4.0f;
            }

            h /= 6.0f;
            if (h < 0) h += 1.0f;
        }

        public static Color ValueToColor(float value, DynamicSettings dynamicSettings, float? bgThresholdOverride = null)
        {
            value *= dynamicSettings.ColorIncreaseFactor;

            if (value > 1.0f && dynamicSettings.ColorOverflow)
            {
                if (((int)value % 2) == 0)
                    value = value % 1.0f;
                else
                    value = 1.0f - (value % 1.0f);
            }

            if (value > 1.0f)
                value = 1.0f;

            List<ColorSetting> colors = dynamicSettings.Colors;

            if (colors.Count == 0)
                return Color.Black;

            if (dynamicSettings.UseRainbow)
                return ValueToRainbowColor(value, dynamicSettings, bgThresholdOverride);

            var backgroundColor = dynamicSettings.Colors[0];
            if (colors.Count < 2)
                return backgroundColor.Color;

            float colorTransition = dynamicSettings.ColorTransition;
            int paletteIdx = 0;
            float sectionEnd = bgThresholdOverride ?? backgroundColor.Threshold;
            Color currColor = backgroundColor.Color;

            for (int i = 0; i < colors.Count; i++)
            {
                var entry = colors[i];
                float threshold = entry.Threshold;
                if (i == 0 && bgThresholdOverride != null)
                    threshold = bgThresholdOverride.Value;

                if (value < threshold || (value == 1.0f && value <= threshold))
                {
                    paletteIdx = i;
                    sectionEnd = threshold;
                    currColor = entry.Color;
                    break;
                }
            }

            float sectionStart = 0f;
            if (paletteIdx != 0)
            {
                if (paletteIdx - 1 == 0 && bgThresholdOverride != null)
                    sectionStart = bgThresholdOverride.Value;
                else
                    sectionStart = colors[paletteIdx - 1].Threshold;
            }
                

            float sectionSize = sectionEnd - sectionStart;
            float transitionAreaSize = 0f;

            bool isAboveHalf = value >= sectionSize / 2f + sectionStart;

            if (isAboveHalf)
            {
                if (paletteIdx == colors.Count - 1)
                    return currColor;

                float nextEnd = colors[paletteIdx + 1].Threshold;
                float nextSize = nextEnd - sectionEnd;

                transitionAreaSize = nextSize > sectionSize
                    ? sectionSize * colorTransition
                    : nextSize * colorTransition;
            }
            else
            {
                if (paletteIdx == 0)
                    return currColor;

                float prevStart = 0f;
                if (paletteIdx > 1)
                {
                    if (paletteIdx - 2 == 0 && bgThresholdOverride != null)
                        prevStart = bgThresholdOverride.Value;
                    else
                        prevStart = colors[paletteIdx - 2].Threshold;
                }

                float prevSize = sectionStart - prevStart;

                transitionAreaSize = prevSize > sectionSize
                    ? sectionSize * colorTransition
                    : prevSize * colorTransition;
            }

            if (isAboveHalf)
            {
                if (value < sectionEnd - transitionAreaSize || transitionAreaSize == 0)
                    return currColor;

                float t = (sectionEnd - value) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(colors[paletteIdx + 1].Color, currColor, 0.5f),
                    currColor,
                    t
                );
            }
            else
            {
                if (value > sectionStart + transitionAreaSize || transitionAreaSize == 0)
                    return currColor;

                float t = (value - sectionStart) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(colors[paletteIdx - 1].Color, currColor, 0.5f),
                    currColor,
                    t
                );
            }
        }

        private static Color ValueToRainbowColor(float value, DynamicSettings dynamicSettings, float? bgThresholdOverride)
        {
            var colorTransition = dynamicSettings.ColorTransition;
            var backgroundEntry = dynamicSettings.Colors[0];
            var backgroundThreshold = bgThresholdOverride ?? backgroundEntry.Threshold;
            float transitionAreaSize = 0f;

            if (backgroundThreshold > 0.5f)
                transitionAreaSize = (1f - backgroundThreshold) * colorTransition;
            else
                transitionAreaSize = backgroundThreshold * colorTransition;

            if (value < backgroundThreshold - transitionAreaSize || backgroundThreshold == 1f)
                return backgroundEntry.Color;

            float t = (value - backgroundThreshold) / (1f - backgroundThreshold);
            float hue = t * 0.75f;
            if (hue < 0f) hue = 0f;

            var color = HsvToRgb(hue, 1f, 1f);

            if (value < backgroundThreshold)
            {
                if (value < backgroundThreshold - transitionAreaSize || transitionAreaSize == 0)
                    return backgroundEntry.Color;

                t = (backgroundThreshold - value) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(color, backgroundEntry.Color, 0.5f),
                    backgroundEntry.Color,
                    t
                );
            }
            else
            {
                if (value > backgroundThreshold + transitionAreaSize || transitionAreaSize == 0)
                    return color;

                t = (value - backgroundThreshold) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(backgroundEntry.Color, color, 0.5f),
                    color,
                    t
                );
            }
        }

        public static Color NonAudioValueToColor(float value, float audioValue, DynamicSettings dynamicSettings)
        {
            float valueToColorBias = dynamicSettings.ValueColorBias;
            var colors = dynamicSettings.Colors;
            Color color;

            if (valueToColorBias <= 0.0f)
            {
                color = ValueToColor(value, dynamicSettings, 0.0f);
            } 
            else if (valueToColorBias >= 1.0f)
            {
                color = ValueToColor(audioValue, dynamicSettings, 0.0f);
            } 
            else
            {
                color = LerpColor(
                    ValueToColor(value, dynamicSettings, 0.0f),
                    ValueToColor(audioValue, dynamicSettings, 0.0f),
                    valueToColorBias
                );
            }

            if (!dynamicSettings.GetAlphaFromValue)
                return color;

            var colorTransition = dynamicSettings.ColorTransition;
            var backgroundEntry = colors[0];

            float transitionAreaSize = backgroundEntry.Threshold > 0.5f
                ? (1f - backgroundEntry.Threshold) * colorTransition
                : backgroundEntry.Threshold * colorTransition;

            if (audioValue < backgroundEntry.Threshold - transitionAreaSize || backgroundEntry.Threshold == 1f)
                return backgroundEntry.Color;

            if (audioValue < backgroundEntry.Threshold)
            {
                if (audioValue < backgroundEntry.Threshold - transitionAreaSize || transitionAreaSize == 0)
                    return backgroundEntry.Color;

                float t = (backgroundEntry.Threshold - audioValue) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(color, backgroundEntry.Color, 0.5f),
                    backgroundEntry.Color,
                    t
                );
            } 
            else
            {
                if (audioValue > backgroundEntry.Threshold + transitionAreaSize || transitionAreaSize == 0)
                    return color;

                float t = (audioValue - backgroundEntry.Threshold) / transitionAreaSize;
                t = MathHelper.Clamp(t, 0f, 1f);

                return LerpColor(
                    LerpColor(backgroundEntry.Color, color, 0.5f),
                    color,
                    t
                );
            }
        }
    }
}
