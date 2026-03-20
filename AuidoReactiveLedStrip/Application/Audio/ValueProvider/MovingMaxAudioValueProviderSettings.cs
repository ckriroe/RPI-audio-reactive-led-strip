namespace Application.Audio.ValueProvider
{
    public class MovingMaxAudioValueProviderSettings : BaseAudioValueProviderSettings
    {
        public int LastExtraOrdanarySampleBufferSize { get; set; }
        public float BelowMinFreqAmplitudeFunctionFactor { get; set; }
        public int MaxFreqAmplitudeIncreaseRatio { get; set; }
        public int MaxFreqAmplitudeDecreaseRatio { get; set; }
        public int MaxFreqAmplitudeTTL { get; set; }
        public float MaxFreqAmplitudeProlongerThreshholdPercent { get; set; }
        public float MaxFreqAmplitudeDecayRate { get; set; }
        public float PercentDiffFromMaxToBeExtraOrdanary { get; set; }
    }
}
