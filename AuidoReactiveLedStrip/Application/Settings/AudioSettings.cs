namespace Application.Settings
{
    public class AudioSettings
    {
        public int SampleRate { get; set; }
        public int Channels { get; set; }
        public int FftSize { get; set; }
        public int AudioDeviceId { get; set; }
    }
}
