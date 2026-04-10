using Application.RuntimeSettings;

namespace Application.Audio.Receiver
{
    public interface IAudioReceiver
    {
        void ApplyStaticSettings(StaticSettings staticSettings);

        void RegisterAudioConsumer(Guid identifier, int requestedBufferSize, Action<float[]> audioCallback);

        void UnregisterAudioConsumer(Guid identifier);
    }
}
