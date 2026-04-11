using Application.Util;
using System.Drawing;

namespace Application.RuntimeSettings
{
    public static class SettingsCorrector
    {
        public static void CorrectDynamicPresetSettings(DynamicPresetSettings dynamicPresetSettings, StaticSettings staticSettings)
        {
            if (dynamicPresetSettings.Presets.Count == 0)
                dynamicPresetSettings.SelectedPresetIndex = -1;

            if (dynamicPresetSettings.SelectedPresetIndex >= dynamicPresetSettings.Presets.Count)
                dynamicPresetSettings.SelectedPresetIndex = dynamicPresetSettings.Presets.Count - 1;

            foreach (Preset preset in dynamicPresetSettings.Presets)
            {
                CorrectDynamicEffectSettings(preset.EffectSettings, staticSettings);
            }
        }

        public static void CorrectStaticSettings(StaticSettings staticSettings)
        {
            staticSettings.Fps = Math.Max(staticSettings.Fps, 1);
            staticSettings.LedCount = Math.Max(staticSettings.LedCount, 2);
        }

        private static void CorrectDynamicEffectSettings(DynamicEffectSettings dynamicSettings, StaticSettings staticSettings)
        {
            dynamicSettings.CalculatedLedCount = Math.Max(2, staticSettings.LedCount / dynamicSettings.EffectRepeats);

            dynamicSettings.EffectOrigin = Math.Min(dynamicSettings.CalculatedLedCount - 1, dynamicSettings.EffectOrigin / dynamicSettings.EffectRepeats);
            dynamicSettings.ColorWaveOrigin = Math.Min(dynamicSettings.CalculatedLedCount - 1, dynamicSettings.ColorWaveOrigin / dynamicSettings.EffectRepeats);
            dynamicSettings.PatternCenter = Math.Min(dynamicSettings.CalculatedLedCount - 1, dynamicSettings.PatternCenter / dynamicSettings.EffectRepeats);

            dynamicSettings.MinFreq = Math.Clamp(dynamicSettings.MinFreq, 0, 20000);
            dynamicSettings.MaxFreq = Math.Clamp(dynamicSettings.MaxFreq, 0, 20000);
            
            dynamicSettings.FftSize = Math.Max(dynamicSettings.FftSize, 1);

            if (dynamicSettings.MaxFreq < dynamicSettings.MinFreq)
                dynamicSettings.MaxFreq = dynamicSettings.MinFreq;

            dynamicSettings.SpectrumSections = Math.Min(dynamicSettings.CalculatedLedCount, dynamicSettings.SpectrumSections);

            List<ColorSetting> correctedColorSettings = [];
            float lastThreshold = 0.0f;
            for (int i = 0; i < dynamicSettings.Colors.Count; i++)
            {
                ColorSetting colorSetting = dynamicSettings.Colors[i];

                if (string.IsNullOrEmpty(colorSetting.Color))
                    colorSetting.ColorInstance = Color.Black;
                else
                    colorSetting.ColorInstance = ColorTranslator.FromHtml(colorSetting.Color);

                colorSetting.Threshold = MathHelper.Clamp(colorSetting.Threshold, 0.0f, 1.0f);
                if (i == dynamicSettings.Colors.Count - 1)
                    colorSetting.Threshold = 1.0f;

                if (colorSetting.Threshold < lastThreshold)
                    continue;
                
                correctedColorSettings.Add(colorSetting);
                lastThreshold = colorSetting.Threshold;
            }

            dynamicSettings.Colors = correctedColorSettings;
            dynamicSettings.MaxFreqAmplitude = 500;
        }
    }
}
