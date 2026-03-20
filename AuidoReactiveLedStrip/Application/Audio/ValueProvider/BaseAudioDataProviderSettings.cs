namespace Application.Audio.ValueProvider
{
    public class BaseAudioDataProviderSettings
    {
        public int SampleRate { get; set; } 
        public int FftSize { get; set; }
        public int MinFrequency { get; set; }
        public int MaxFrequency { get; set; }
    }
}
