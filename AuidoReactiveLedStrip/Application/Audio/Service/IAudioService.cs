using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Audio.Service
{
    public interface IAudioService
    {
        float[]? GetCurrentFftData();
        float? GetCurrentAudioValue();
        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);
        void SetAudioMode(AudioServiceMode audioMode);
    }
}