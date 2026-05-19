using Application.Domain;

namespace Application.RuntimeSettings
{
    public class TemplateEffectSettings
    {
        public List<ColorSetting>? Colors { get; set; }

        public bool? UseRainbow { get; set; }

        public int? EffectOrigin { get; set; }

        public int? Speed { get; set; }

        public int? MinFreq { get; set; }

        public int? MaxFreq { get; set; }

        public float? Fade { get; set; }

        public float? FadeOverTime { get; set; }

        public bool? StaticSpectrum { get; set; }

        public int? SpectrumSections { get; set; }

        public bool? BouncyWave { get; set; }

        public float? Saturate { get; set; }

        public float? SaturateThreshold { get; set; }

        public int? MeanValueBufferSize { get; set; }

        public float? MeanValueThreshold { get; set; }

        public AudioMode? AudioMode { get; set; }

        public EffectMode? EffectMode { get; set; }

        public ColorMode? ColorMode { get; set; }

        public float? MinFreqAmplitude { get; set; }

        public float? MaxFreqAmplitude { get; set; }

        public float? ColorIncreaseFactor { get; set; }

        public float? ValueIncreaseFactor { get; set; }

        public float? ColorTransition { get; set; }

        public float? ValueColorBias { get; set; }

        public int? ColorWaveOrigin { get; set; }

        public int? ColorWaveSpeed { get; set; }

        public int? ColorWaveSize { get; set; }

        public bool? ColorWaveInwards { get; set; }

        public float? NoiseAmount { get; set; }

        public float? NoiseSmoothing { get; set; }

        public bool? GetAlphaFromValue { get; set; }

        public bool? ColorOverflow { get; set; }

        public float? Brightness { get; set; }

        public float? Gamma { get; set; }

        public int? EffectRepeats { get; set; }

        public float? Acceleration { get; set; }

        public int? ParticleSize { get; set; }

        public int? PatternSplits { get; set; }

        public int? PatternSpread { get; set; }

        public int? PatternFlip { get; set; }

        public int? PatternCenter { get; set; }
        
        public float? PatternSectionSizeMod { get; set; }

        public int? FftSize { get; set; }

        public int? BpmLimit { get; set; }

        public float? AudioResponseCurve { get; set; }

        public int? AudioPeakHoldTimeMs { get; set; }

        public int? CalculatedLedCount { get; set; }

        public Guid? NextEffectId { get; set; }

        public long? EffectDurationMs { get; set; }

        public long? EffectTransitionDurationMs { get; set; }

        public long? EffectTransitionWarmupDuration { get; set; }

        public bool? ResetEffectAfterTransition { get; set; }

        public float? EnergyInfluence { get; set; }
        
        public float? FluxInfluence { get; set; }

        public Guid? TemplateId { get; set; }
    }
}
