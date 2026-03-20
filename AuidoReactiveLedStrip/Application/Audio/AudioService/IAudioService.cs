using Application.Domain;

namespace Application.Audio.AudioService
{
    public interface IAudioService
    {
        void SetAudioMode(AudioServiceMode audioMode);
        float[]? GetCurrentFftData();
        float? GetCurrentAudioValue();
    }
}