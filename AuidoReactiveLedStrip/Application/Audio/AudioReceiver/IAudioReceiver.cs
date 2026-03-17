namespace Application.Audio.AudioReceiver
{
    public interface IAudioReceiver
    {
        void Initialize(int audioDeviceId, int channelCount, int bufferSize, int sampleRate);

        void StartAudioStream(Action<float[]> audioCallback);

        void StopAudioStream();
    }
}
