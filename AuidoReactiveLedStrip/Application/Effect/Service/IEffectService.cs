using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Effect.Service
{
    public interface IEffectService
    {
        LedStrip? GetLedStrip();

        void ApplySettings(IAudioService affectedAudioService, StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);

        void Reset();
    }
}