using Application.Domain;

namespace Application.RuntimeSettings
{
    public class StaticSettings
    {
        public int SampleRate { get; set; }
        public int Channels { get; set; }
        public int AudioDeviceId { get; set; }
        public float BelowMinFreqAmplitudeFunctionFactor { get; set; }
        public float MaxFreqAmplitudeIncreaseRatio { get; set; }
        public float MaxFreqAmplitudeDecreaseRatio { get; set; }
        public int MaxFreqAmplitudeTTL { get; set; }
        public float MaxFreqAmplitudeProlongerThreshholdPercent { get; set; }
        public float MaxFreqAmplitudeDecayRate { get; set; }
        public int LedUpdateFrequency { get; set; }
        public int BounceLayers { get; set; }
        public int MaxEffectSpeed { get; set; }
        public int ExternalModeRelayGpio { get; set; }
        public int InvalidFrameSleepTime { get; set; }
        public bool PrintFrameTimes { get; set; }
        public bool PrintFrequencyInfos { get; set; }
        public bool PrintSequenceInfos { get; set; }
        public float MinSanitizedValue {  get; set; }
        public bool UseGuiVisualization { get; set; }
        public bool UseLedVisualization { get; set; }
        public int GuiWidth { get; set; }
        public int GuiHeight { get; set; }
        public int MainThreadSettingsCheckIntervalMs { get; set; }
        public bool RunIndefinitely { get; set; }
        public bool OutputWhenGpioOff { get; set; }
        public bool AccurateSleeping { get; set; }
        public GuiVisualizationMode GuiVisualizationMode { get; set; }
        public int RectangleGuiVisualizationHeight { get; set; }
        public int Fps { get; set; }
        public int LedCount { get; set; }
    }
}
