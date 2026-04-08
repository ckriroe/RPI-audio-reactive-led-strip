using Application.Util;
using System.Drawing;

namespace Application.RuntimeSettings
{
    public static class SettingsCorrector
    {
        public static void CorrectDynamicSettings(DynamicSettings dynamicSettings)
        {
            dynamicSettings.PhysicalLedCount = dynamicSettings.LedCount;
            dynamicSettings.LedCount = Math.Max(2, dynamicSettings.PhysicalLedCount / dynamicSettings.EffectRepeats);
            dynamicSettings.EffectOrigin = Math.Min(dynamicSettings.LedCount, dynamicSettings.EffectOrigin / dynamicSettings.EffectRepeats);
            dynamicSettings.ColorWaveOrigin = Math.Min(dynamicSettings.LedCount, dynamicSettings.ColorWaveOrigin / dynamicSettings.EffectRepeats);
            dynamicSettings.PatternCenter = Math.Min(dynamicSettings.LedCount, dynamicSettings.PatternCenter / dynamicSettings.EffectRepeats);
            dynamicSettings.MinFreq = Math.Clamp(dynamicSettings.MinFreq, 0, 20000);
            dynamicSettings.MaxFreq = Math.Clamp(dynamicSettings.MaxFreq, 0, 20000);
            dynamicSettings.Fps = Math.Max(dynamicSettings.Fps, 1);
            dynamicSettings.FftSize = Math.Max(dynamicSettings.FftSize, 1);

            if (dynamicSettings.MaxFreq < dynamicSettings.MinFreq)
                dynamicSettings.MaxFreq = dynamicSettings.MinFreq;

            dynamicSettings.SpectrumSections = Math.Min(dynamicSettings.LedCount, dynamicSettings.SpectrumSections);

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
