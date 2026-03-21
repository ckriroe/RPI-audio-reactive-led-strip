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
            int finalR = (int)Math.Min(255.0, ColorHelper.GammaCorrect(led.Color.R * dynamicSettings.Brightness, dynamicSettings));
            int finalG = (int)Math.Min(255.0, ColorHelper.GammaCorrect(led.Color.G * dynamicSettings.Brightness, dynamicSettings));
            int finalB = (int)Math.Min(255.0, ColorHelper.GammaCorrect(led.Color.B * dynamicSettings.Brightness, dynamicSettings));

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
