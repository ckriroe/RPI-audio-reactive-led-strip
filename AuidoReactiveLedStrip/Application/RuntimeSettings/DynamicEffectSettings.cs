using Application.Domain;
using System.Drawing;

namespace Application.RuntimeSettings
{
    public class DynamicEffectSettings
    {
        public List<ColorSetting> Colors { get; set; } = [];

        public bool UseRainbow { get; set; }

        public int EffectOrigin { get; set; }

        public int Speed { get; set; }

        public int MinFreq { get; set; }

        public int MaxFreq { get; set; }

        public float Fade { get; set; }

        public float FadeOverTime { get; set; }

        public bool StaticSpectrum { get; set; }

        public int SpectrumSections { get; set; }

        public bool BouncyWave { get; set; }

        public float Saturate { get; set; }

        public float SaturateThreshold { get; set; }

        public int MeanValueBufferSize { get; set; }

        public float MeanValueThreshold { get; set; }

        public AudioMode AudioMode { get; set; }

        public EffectMode EffectMode { get; set; }

        public ColorMode ColorMode { get; set; }

        public float MinFreqAmplitude { get; set; }

        public float MaxFreqAmplitude { get; set; }

        public float ColorIncreaseFactor { get; set; }

        public float ValueIncreaseFactor { get; set; }

        public float ColorTransition { get; set; }

        public float ValueColorBias { get; set; }

        public int ColorWaveOrigin { get; set; }

        public int ColorWaveSpeed { get; set; }

        public int ColorWaveSize { get; set; }

        public bool ColorWaveInwards { get; set; }

        public float NoiseAmount { get; set; }

        public float NoiseSmoothing { get; set; }

        public bool GetAlphaFromValue { get; set; }

        public bool ColorOverflow { get; set; }

        public float Brightness { get; set; }

        public float Gamma { get; set; }

        public int EffectRepeats { get; set; }

        public float Acceleration { get; set; }

        public int ParticleSize { get; set; }

        public int PatternSplits { get; set; }

        public int PatternSpread { get; set; }

        public int PatternFlip { get; set; }

        public int PatternCenter { get; set; }
        
        public float PatternSectionSizeMod { get; set; }

        public int FftSize { get; set; }

        public int BpmLimit { get; set; }

        public float AudioResponseCurve { get; set; }

        public int AudioPeakHoldTimeMs { get; set; }

        public int CalculatedLedCount { get; set; }

        public Guid? NextEffectId { get; set; }

        public long EffectDurationMs { get; set; }

        public long EffectTransitionDurationMs { get; set; }

        public long EffectTransitionWarmupDuration { get; set; }

        public bool ResetEffectAfterTransition { get; set; }

        public float EnergyInfluence { get; set; }
        
        public float FluxInfluence { get; set; }

        public override bool Equals(object? obj)
        {
            return obj is DynamicEffectSettings settings &&
                   this.Colors.SequenceEqual(settings.Colors) &&
                   this.UseRainbow == settings.UseRainbow &&
                   this.EffectOrigin == settings.EffectOrigin &&
                   this.Speed == settings.Speed &&
                   this.MinFreq == settings.MinFreq &&
                   this.MaxFreq == settings.MaxFreq &&
                   this.Fade == settings.Fade &&
                   this.FadeOverTime == settings.FadeOverTime &&
                   this.StaticSpectrum == settings.StaticSpectrum &&
                   this.SpectrumSections == settings.SpectrumSections &&
                   this.BouncyWave == settings.BouncyWave &&
                   this.Saturate == settings.Saturate &&
                   this.SaturateThreshold == settings.SaturateThreshold &&
                   this.MeanValueBufferSize == settings.MeanValueBufferSize &&
                   this.MeanValueThreshold == settings.MeanValueThreshold &&
                   this.AudioMode == settings.AudioMode &&
                   this.EffectMode == settings.EffectMode &&
                   this.ColorMode == settings.ColorMode &&
                   this.MinFreqAmplitude == settings.MinFreqAmplitude &&
                   this.MaxFreqAmplitude == settings.MaxFreqAmplitude &&
                   this.ColorIncreaseFactor == settings.ColorIncreaseFactor &&
                   this.ValueIncreaseFactor == settings.ValueIncreaseFactor &&
                   this.ColorTransition == settings.ColorTransition &&
                   this.ValueColorBias == settings.ValueColorBias &&
                   this.ColorWaveOrigin == settings.ColorWaveOrigin &&
                   this.ColorWaveSpeed == settings.ColorWaveSpeed &&
                   this.ColorWaveSize == settings.ColorWaveSize &&
                   this.ColorWaveInwards == settings.ColorWaveInwards &&
                   this.NoiseAmount == settings.NoiseAmount &&
                   this.NoiseSmoothing == settings.NoiseSmoothing &&
                   this.GetAlphaFromValue == settings.GetAlphaFromValue &&
                   this.ColorOverflow == settings.ColorOverflow &&
                   this.Brightness == settings.Brightness &&
                   this.Gamma == settings.Gamma &&
                   this.EffectRepeats == settings.EffectRepeats &&
                   this.Acceleration == settings.Acceleration &&
                   this.ParticleSize == settings.ParticleSize &&
                   this.PatternSplits == settings.PatternSplits &&
                   this.PatternSpread == settings.PatternSpread &&
                   this.PatternFlip == settings.PatternFlip &&
                   this.PatternCenter == settings.PatternCenter &&
                   this.PatternSectionSizeMod == settings.PatternSectionSizeMod &&
                   this.FftSize == settings.FftSize &&
                   this.BpmLimit == settings.BpmLimit &&
                   this.AudioResponseCurve == settings.AudioResponseCurve &&
                   this.AudioPeakHoldTimeMs == settings.AudioPeakHoldTimeMs &&
                   this.CalculatedLedCount == settings.CalculatedLedCount &&
                   EqualityComparer<Guid?>.Default.Equals(this.NextEffectId, settings.NextEffectId) &&
                   this.EffectDurationMs == settings.EffectDurationMs &&
                   this.EffectTransitionDurationMs == settings.EffectTransitionDurationMs &&
                   this.EffectTransitionWarmupDuration == settings.EffectTransitionWarmupDuration &&
                   this.ResetEffectAfterTransition == settings.ResetEffectAfterTransition &&
                   this.EnergyInfluence == settings.EnergyInfluence &&
                   this.FluxInfluence == settings.FluxInfluence;
        }

        public override int GetHashCode()
        {
            HashCode hash = new HashCode();
            foreach (ColorSetting color in this.Colors)
            {
                hash.Add(color);
            }

            hash.Add(this.UseRainbow);
            hash.Add(this.EffectOrigin);
            hash.Add(this.Speed);
            hash.Add(this.MinFreq);
            hash.Add(this.MaxFreq);
            hash.Add(this.Fade);
            hash.Add(this.FadeOverTime);
            hash.Add(this.StaticSpectrum);
            hash.Add(this.SpectrumSections);
            hash.Add(this.BouncyWave);
            hash.Add(this.Saturate);
            hash.Add(this.SaturateThreshold);
            hash.Add(this.MeanValueBufferSize);
            hash.Add(this.MeanValueThreshold);
            hash.Add(this.AudioMode);
            hash.Add(this.EffectMode);
            hash.Add(this.ColorMode);
            hash.Add(this.MinFreqAmplitude);
            hash.Add(this.MaxFreqAmplitude);
            hash.Add(this.ColorIncreaseFactor);
            hash.Add(this.ValueIncreaseFactor);
            hash.Add(this.ColorTransition);
            hash.Add(this.ValueColorBias);
            hash.Add(this.ColorWaveOrigin);
            hash.Add(this.ColorWaveSpeed);
            hash.Add(this.ColorWaveSize);
            hash.Add(this.ColorWaveInwards);
            hash.Add(this.NoiseAmount);
            hash.Add(this.NoiseSmoothing);
            hash.Add(this.GetAlphaFromValue);
            hash.Add(this.ColorOverflow);
            hash.Add(this.Brightness);
            hash.Add(this.Gamma);
            hash.Add(this.EffectRepeats);
            hash.Add(this.Acceleration);
            hash.Add(this.ParticleSize);
            hash.Add(this.PatternSplits);
            hash.Add(this.PatternSpread);
            hash.Add(this.PatternFlip);
            hash.Add(this.PatternCenter);
            hash.Add(this.PatternSectionSizeMod);
            hash.Add(this.FftSize);
            hash.Add(this.BpmLimit);
            hash.Add(this.AudioResponseCurve);
            hash.Add(this.AudioPeakHoldTimeMs);
            hash.Add(this.CalculatedLedCount);
            hash.Add(this.NextEffectId);
            hash.Add(this.EffectDurationMs);
            hash.Add(this.EffectTransitionDurationMs);
            hash.Add(this.EffectTransitionWarmupDuration);
            hash.Add(this.ResetEffectAfterTransition);
            hash.Add(this.EnergyInfluence);
            hash.Add(this.FluxInfluence);

            return hash.ToHashCode();
        }
    }

    public class ColorSetting
    {
        public required string Color { get; set; }

        public Color ColorInstance { get; set; }

        public required float Threshold { get; set; }

        public override bool Equals(object? obj)
        {
            return obj is ColorSetting setting &&
                   this.ColorInstance.Equals(setting.ColorInstance) &&
                   this.Threshold == setting.Threshold;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(this.ColorInstance, this.Threshold);
        }
    }
}
