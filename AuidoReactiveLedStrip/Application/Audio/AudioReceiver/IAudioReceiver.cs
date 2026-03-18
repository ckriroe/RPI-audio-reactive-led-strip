namespace Application.Audio.AudioReceiver
{
    public interface IAudioReceiver
    {
        void Initialize(int audioDeviceId, int channelCount, int bufferSize, int sampleRate, Action? initCallback = null);

        void StartAudioStream(Action<float[]> audioCallback);

        void StopAudioStream();
    }
}
