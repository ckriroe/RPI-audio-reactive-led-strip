namespace Application.Settings
{
    public class StaticSettings
    {
        public int SampleRate { get; set; }
        public int Channels { get; set; }
        public int FftSize { get; set; }
        public int AudioDeviceId { get; set; }
        public int LastExtraOrdanarySampleBufferSize { get; set; }
        public float BelowMinFreqAmplitudeFunctionFactor { get; set; }
        public int MaxFreqAmplitudeIncreaseRatio { get; set; }
        public int MaxFreqAmplitudeDecreaseRatio { get; set; }
        public int MaxFreqAmplitudeTTL { get; set; }
        public float MaxFreqAmplitudeProlongerThreshholdPercent { get; set; }
        public float MaxFreqAmplitudeDecayRate { get; set; }
        public float PercentDiffFromMaxToBeExtraOrdanary { get; set; }
        public int Fps { get; set; }
        public int BounceLayers { get; set; }
        public int MaxEffectSpeed { get; set; }
        public int ExternalModeRelayGpio { get; set; }
        public int InvalidFrameSleepTime { get; set; }
        public int ReloadSettingsAfterMs { get; set; }
        public bool PrintFrameTimes { get; set; }
    }
}
