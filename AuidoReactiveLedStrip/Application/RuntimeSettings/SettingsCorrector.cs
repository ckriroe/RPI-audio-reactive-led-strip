using Application.Util;
using System.Drawing;
using System.Runtime.CompilerServices;

namespace Application.RuntimeSettings
{
    public static class SettingsCorrector
    {
        public static void CorrectDynamicPresetSettings(DynamicPresetSettings dynamicPresetSettings, StaticSettings staticSettings)
        {
            foreach (Preset preset in dynamicPresetSettings.Presets)
            {
                preset.EffectSettings = ResolveEffectSettings(dynamicPresetSettings, preset.TemplateEffectSettings);
                CorrectDynamicEffectSettings(preset.EffectSettings, staticSettings);
            }
        }

        private static DynamicEffectSettings ResolveEffectSettings(DynamicPresetSettings dynamicPresetSettings, TemplateEffectSettings tes)
        {
            TemplateEffectSettings? temp = tes.TemplateId != null ?
                dynamicPresetSettings.Presets.FirstOrDefault(p => p.Id == tes.TemplateId)?.TemplateEffectSettings
                : null;

            return new DynamicEffectSettings()
            {
                Colors                              = tes.Colors ?? temp?.Colors ?? [],
                UseRainbow                          = GetTemplateVal(tes, temp, t => t.UseRainbow),
                EffectOrigin                        = GetTemplateVal(tes, temp, t => t.EffectOrigin),
                Speed                               = GetTemplateVal(tes, temp, t => t.Speed),
                MinFreq                             = GetTemplateVal(tes, temp, t => t.MinFreq),
                MaxFreq                             = GetTemplateVal(tes, temp, t => t.MaxFreq),
                Fade                                = GetTemplateVal(tes, temp, t => t.Fade),
                FadeOverTime                        = GetTemplateVal(tes, temp, t => t.FadeOverTime),
                StaticSpectrum                      = GetTemplateVal(tes, temp, t => t.StaticSpectrum),
                SpectrumSections                    = GetTemplateVal(tes, temp, t => t.SpectrumSections),
                BouncyWave                          = GetTemplateVal(tes, temp, t => t.BouncyWave),
                Saturate                            = GetTemplateVal(tes, temp, t => t.Saturate),
                SaturateThreshold                   = GetTemplateVal(tes, temp, t => t.SaturateThreshold),
                MeanValueBufferSize                 = GetTemplateVal(tes, temp, t => t.MeanValueBufferSize),
                MeanValueThreshold                  = GetTemplateVal(tes, temp, t => t.MeanValueThreshold),
                AudioMode                           = GetTemplateVal(tes, temp, t => t.AudioMode),
                EffectMode                          = GetTemplateVal(tes, temp, t => t.EffectMode),
                ColorMode                           = GetTemplateVal(tes, temp, t => t.ColorMode),
                MinFreqAmplitude                    = GetTemplateVal(tes, temp, t => t.MinFreqAmplitude),
                MaxFreqAmplitude                    = GetTemplateVal(tes, temp, t => t.MaxFreqAmplitude),
                ColorIncreaseFactor                 = GetTemplateVal(tes, temp, t => t.ColorIncreaseFactor),
                ValueIncreaseFactor                 = GetTemplateVal(tes, temp, t => t.ValueIncreaseFactor),
                ColorTransition                     = GetTemplateVal(tes, temp, t => t.ColorTransition),
                ValueColorBias                      = GetTemplateVal(tes, temp, t => t.ValueColorBias),
                ColorWaveOrigin                     = GetTemplateVal(tes, temp, t => t.ColorWaveOrigin),
                ColorWaveSpeed                      = GetTemplateVal(tes, temp, t => t.ColorWaveSpeed),
                ColorWaveSize                       = GetTemplateVal(tes, temp, t => t.ColorWaveSize),
                ColorWaveInwards                    = GetTemplateVal(tes, temp, t => t.ColorWaveInwards),
                NoiseAmount                         = GetTemplateVal(tes, temp, t => t.NoiseAmount),
                NoiseSmoothing                      = GetTemplateVal(tes, temp, t => t.NoiseSmoothing),
                GetAlphaFromValue                   = GetTemplateVal(tes, temp, t => t.GetAlphaFromValue),
                ColorOverflow                       = GetTemplateVal(tes, temp, t => t.ColorOverflow),
                Brightness                          = GetTemplateVal(tes, temp, t => t.Brightness),
                Gamma                               = GetTemplateVal(tes, temp, t => t.Gamma),
                EffectRepeats                       = GetTemplateVal(tes, temp, t => t.EffectRepeats, 1),
                Acceleration                        = GetTemplateVal(tes, temp, t => t.Acceleration),
                ParticleSize                        = GetTemplateVal(tes, temp, t => t.ParticleSize),
                PatternSplits                       = GetTemplateVal(tes, temp, t => t.PatternSplits),
                PatternSpread                       = GetTemplateVal(tes, temp, t => t.PatternSpread),
                PatternFlip                         = GetTemplateVal(tes, temp, t => t.PatternFlip),
                PatternCenter                       = GetTemplateVal(tes, temp, t => t.PatternCenter),
                PatternSectionSizeMod               = GetTemplateVal(tes, temp, t => t.PatternSectionSizeMod),
                FftSize                             = GetTemplateVal(tes, temp, t => t.FftSize),
                BpmLimit                            = GetTemplateVal(tes, temp, t => t.BpmLimit, -1),
                AudioResponseCurve                  = GetTemplateVal(tes, temp, t => t.AudioResponseCurve, 1.0f),
                AudioPeakHoldTimeMs                 = GetTemplateVal(tes, temp, t => t.AudioPeakHoldTimeMs),
                CalculatedLedCount                  = GetTemplateVal(tes, temp, t => t.CalculatedLedCount),
                NextEffectId                        = tes.NextEffectId,
                EffectDurationMs                    = GetTemplateVal(tes, temp, t => t.EffectDurationMs),
                EffectTransitionDurationMs          = GetTemplateVal(tes, temp, t => t.EffectTransitionDurationMs),
                EffectTransitionWarmupDuration      = GetTemplateVal(tes, temp, t => t.EffectTransitionWarmupDuration),
                ResetEffectAfterTransition          = GetTemplateVal(tes, temp, t => t.ResetEffectAfterTransition),
                EnergyInfluence                     = GetTemplateVal(tes, temp, t => t.EnergyInfluence),
                FluxInfluence                       = GetTemplateVal(tes, temp, t => t.FluxInfluence)
            };
        }

        private static T GetTemplateVal<T>(TemplateEffectSettings effectSettings, TemplateEffectSettings? template, Func<TemplateEffectSettings, T?> valueFunc, T defaultValue = default) where T : struct
        {
            T? val = valueFunc(effectSettings);
            if (val != null)
                return val.Value;

            if (template != null)
                val = valueFunc(template);

            return val ?? defaultValue;
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
