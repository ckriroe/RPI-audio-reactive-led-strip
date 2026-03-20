using Application.Domain;

namespace Application.Audio.Service
{
    public interface IAudioService
    {
        void SetAudioMode(AudioServiceMode audioMode);
        float[]? GetCurrentFftData();
        float? GetCurrentAudioValue();
    }
}