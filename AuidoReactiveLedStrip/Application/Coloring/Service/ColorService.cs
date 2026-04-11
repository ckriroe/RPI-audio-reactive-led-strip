using Application.Coloring.ColorCorrection;
using Application.Coloring.Mode;
using Application.Coloring.Sanitizing;
using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Service
{
    public class ColorService : IColorService
    {
        private readonly IValueSanitizer valueSanitizer;
        private readonly ValueColorMode valueColorMode;
        private readonly IndexColorMode indexColorMode;
        private readonly DistanceToCenterColorMode distanceToCenterColorMode;
        private readonly DistanceToBorderColorMode distanceToBorderColorMode;
        private readonly ColorWaveColorMode colorWaveColorMode;
        private readonly ColorIslandColorMode colorIslandColorMode;

        private readonly GammaCorrector gammaCorrector;
        private readonly BrightnessAdjuster brightnessAdjuster;

        private ColorMode? currentColorModeType = null;
        private IColorMode? currentColorMode = null;

        private StaticSettings? staticSettings = null;
        private DynamicEffectSettings? dynamicSettings = null;

        public ColorService(
            IValueSanitizer valueSanitizer,
            ValueColorMode valueColorMode,
            IndexColorMode indexColorMode,
            DistanceToCenterColorMode distanceToCenterColorMode,
            DistanceToBorderColorMode distanceToBorderColorMode,
            ColorWaveColorMode colorWaveColorMode,
            ColorIslandColorMode colorIslandColorMode,
            GammaCorrector gammaCorrector,
            BrightnessAdjuster brightnessAdjuster
        )
        {
            this.valueSanitizer = valueSanitizer;
            this.valueColorMode = valueColorMode;
            this.indexColorMode = indexColorMode;
            this.distanceToCenterColorMode = distanceToCenterColorMode;
            this.distanceToBorderColorMode = distanceToBorderColorMode;
            this.colorWaveColorMode = colorWaveColorMode;
            this.colorIslandColorMode = colorIslandColorMode;
            this.gammaCorrector = gammaCorrector;
            this.brightnessAdjuster = brightnessAdjuster;
        }

        public void ColorizeLedStrip(LedStrip ledStrip)
        {
            LedPixel[] pixels = ledStrip.LedPixels;
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            StaticSettings? staticSettings = this.staticSettings;

            if (pixels.Length < 2 ||
                dynamicSettings == null ||
                staticSettings == null ||
                dynamicSettings.Colors.Count < 2 ||
                this.currentColorMode == null)
            {
                return;
            }

            this.currentColorMode.PrecomputeValues(staticSettings, dynamicSettings, ledStrip);
            int length = pixels.Length;
            var bgColor = dynamicSettings.Colors[0].ColorInstance;

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

        public void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;
            this.SetColorMode(dynamicSettings.ColorMode);
        }

        private void ColorCorrect(LedPixel led, DynamicEffectSettings dynamicSettings)
        {
            (float R, float G, float B) result = this.gammaCorrector.ColorCorrect(this.brightnessAdjuster.ColorCorrect(
                (led.Color.R, led.Color.G, led.Color.B),
                dynamicSettings
            ), dynamicSettings);

            led.Color = Color.FromArgb(
                (int)Math.Clamp(result.R, 0, 255),
                (int)Math.Clamp(result.G, 0, 255),
                (int)Math.Clamp(result.B, 0, 255)
            );
        }

        private void SetColorMode(ColorMode colorMode)
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
