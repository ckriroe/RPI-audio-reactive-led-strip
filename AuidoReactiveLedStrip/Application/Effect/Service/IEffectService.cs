using Application.Domain;

namespace Application.Effect.Service
{
    public interface IEffectService
    {
        LedStrip? GetRenderedLedStrip();
        void SetEffectMode(EffectMode effectMode);
        AudioServiceMode GetRequiredAudioMode();
    }
}