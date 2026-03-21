using Application.Coloring.Mode;
using Application.Coloring.Noise;
using Application.Coloring.Sanitizing;
using Application.Domain;
using Application.Settings;
using Application.Util;
using Microsoft.Extensions.Options;
using rpi_ws281x;
using System;
using System.Drawing;

namespace Application.Coloring.Service
{
    public class ColorService : IColorService
    {
        private readonly IOptionsMonitor<DynamicSettings> dyamicSettings;
        private readonly IOptionsMonitor<StaticSettings> staticSettings;
        private readonly IValueSanitizer valueSanitizer;
        private readonly ValueColorMode valueColorMode;
        private readonly IndexColorMode indexColorMode;
        private readonly DistanceToCenterColorMode distanceToCenterColorMode;
        private readonly DistanceToBorderColorMode distanceToBorderColorMode;
        private readonly ColorWaveColorMode colorWaveColorMode;
        private readonly ColorIslandColorMode colorIslandColorMode;

        private ColorMode? currentColorModeType = null;
        private IColorMode? currentColorMode = null;

        public ColorService(
            IOptionsMonitor<DynamicSettings> dyamicSettings,
            IOptionsMonitor<StaticSettings> staticSettings,
            IValueSanitizer valueSanitizer,
            ValueColorMode valueColorMode,
            IndexColorMode indexColorMode,
            DistanceToCenterColorMode distanceToCenterColorMode,
            DistanceToBorderColorMode distanceToBorderColorMode,
            ColorWaveColorMode colorWaveColorMode,
            ColorIslandColorMode colorIslandColorMode
        )
        {
            this.dyamicSettings = dyamicSettings;
            this.staticSettings = staticSettings;
            this.valueSanitizer = valueSanitizer;
            this.valueColorMode = valueColorMode;
            this.indexColorMode = indexColorMode;
            this.distanceToCenterColorMode = distanceToCenterColorMode;
            this.distanceToBorderColorMode = distanceToBorderColorMode;
            this.colorWaveColorMode = colorWaveColorMode;
            this.colorIslandColorMode = colorIslandColorMode;
        }

        public void ColorizeLedStrip(LedStrip ledStrip)
        {
            IList<LedPixel> pixels = ledStrip.LedPixels;
            DynamicSettings dynamicSettings = this.dyamicSettings.CurrentValue;
            StaticSettings staticSettings = this.staticSettings.CurrentValue;

            if (pixels.Count < 2 || dynamicSettings.Colors.Count < 2 || this.currentColorMode == null)
            {
                return;
            }

            this.currentColorMode.PrecomputeValues(staticSettings, dynamicSettings, ledStrip);
            int length = pixels.Count;
            var bgColor = dynamicSettings.Colors[0].Color;

            for (int i = 0; i < length; i++)
            {
                LedPixel currPixel = pixels[i];
                float sanitizedValue = this.valueSanitizer.SanitizeValue(currPixel.Value, staticSettings);

                if (sanitizedValue == 0.0f)
                {
                    currPixel.Color = bgColor;
                }
                else
                {
                    currPixel.Color = this.currentColorMode.GetColorForValue(dynamicSettings, sanitizedValue, i, length);
                    this.ColorCorrect(currPixel, dynamicSettings);
                }
            }
        }

        private void ColorCorrect(LedPixel led, DynamicSettings dynamicSettings)
        {
            float inv255 = 1.0f / 255.0f;

            float rc = dynamicSettings.RedCorr * 255.0f;
            float gc = dynamicSettings.GreenCorr * 255.0f;
            float bc = dynamicSettings.BlueCorr * 255.0f;
            float hc = dynamicSettings.HueCorr * 360.0f;
            float sc = dynamicSettings.SatCorr;
            float vc = dynamicSettings.ValCorr;
            float rt = dynamicSettings.RedThresh;
            float gt = dynamicSettings.GreenThresh;
            float bt = dynamicSettings.BlueThresh;
            float ht = dynamicSettings.HueThresh;
            float st = dynamicSettings.SatThresh;
            float vt = dynamicSettings.ValThresh;

            float r = led.Color.R;
            float g = led.Color.G;
            float b = led.Color.B;
            float pv = led.Value;

            float rf = rt < 1.0f
                ? MathHelper.Clamp((pv - rt) / (1.0f - rt), 0.0f, 1.0f)
                : 0.0f;

            float gf = gt < 1.0f
                ? MathHelper.Clamp((pv - gt) / (1.0f - gt), 0.0f, 1.0f)
                : 0.0f;

            float bf = bt < 1.0f
                ? MathHelper.Clamp((pv - bt) / (1.0f - bt), 0.0f, 1.0f)
                : 0.0f;

            r += rc * rf;
            g += gc * gf;
            b += bc * bf;

            r = MathHelper.Clamp(r, 0.0f, 255.0f);
            g = MathHelper.Clamp(g, 0.0f, 255.0f);
            b = MathHelper.Clamp(b, 0.0f, 255.0f);

            ColorHelper.RgbToHsv(r * inv255, g * inv255, b * inv255, out float h, out float s, out float v);
            h *= 360.0f;

            float hf = ht < 1.0f
                ? MathHelper.Clamp((pv - ht) / (1.0f - ht), 0.0f, 1.0f)
                : 0.0f;

            float sf = st < 1.0f
                ? MathHelper.Clamp((pv - st) / (1.0f - st), 0.0f, 1.0f)
                : 0.0f;

            float vf = vt < 1.0f
                ? MathHelper.Clamp((pv - vt) / (1.0f - vt), 0.0f, 1.0f)
                : 0.0f;

            h += hc * hf;
            s += sc * sf;
            v += vc * vf;

            h = MathHelper.Clamp(h, 0.0f, 360.0f);
            s = MathHelper.Clamp(s, 0.0f, 1.0f);
            v = MathHelper.Clamp(v, 0.0f, 1.0f);
            ColorHelper.HsvToRgb(h / 360.0f, s, v, out r, out g, out b);

            r *= 255.0f;
            g *= 255.0f;
            b *= 255.0f;

            int finalR = (int)Math.Min(255.0, ColorHelper.GammaCorrect(r * dynamicSettings.Brightness, dynamicSettings));
            int finalG = (int)Math.Min(255.0, ColorHelper.GammaCorrect(g * dynamicSettings.Brightness, dynamicSettings));
            int finalB = (int)Math.Min(255.0, ColorHelper.GammaCorrect(b * dynamicSettings.Brightness, dynamicSettings));

            led.Color = Color.FromArgb(finalR, finalG, finalB);
        }

        public void SetColorMode(ColorMode colorMode)
        {
            if (this.currentColorModeType == colorMode)
                return;

            this.currentColorModeType = colorMode;
            switch (colorMode)
            {
                case ColorMode.ColorByEffectValue:
                    this.currentColorMode = this.valueColorMode;
                    break;
                case ColorMode.ColorByIndex:
                    this.currentColorMode = this.indexColorMode;
                    break;
                case ColorMode.ColorByDistanceToCenter:
                    this.currentColorMode = this.distanceToCenterColorMode;
                    break;
                case ColorMode.ColorByDistanceToBorder:
                    this.currentColorMode = this.distanceToBorderColorMode;
                    break;
                case ColorMode.ColorWave:
                    this.currentColorMode = this.colorWaveColorMode;
                    break;
                case ColorMode.ColorIslands:
                    this.currentColorMode = this.colorIslandColorMode;
                    break;
            }
        }
    }
}
