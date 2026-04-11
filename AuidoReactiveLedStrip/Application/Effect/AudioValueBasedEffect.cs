using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Effect
{
    public abstract class AudioValueBasedEffect : IEffect
    {
        private IAudioService? audioService = null;
        protected DynamicEffectSettings? dynamicEffectSettings = null;
        protected StaticSettings? staticSettings = null;

        protected float GetCurrentAudioValue()
        {
            float valueIncreaseFactor = this.dynamicEffectSettings?.ValueColorBias ?? 0.0f;
            return (this.audioService?.GetCurrentAudioValue() ?? 0.0f) * valueIncreaseFactor;
        }

        public abstract LedStrip? RenderEffekt(LedStrip? prevStrip, int length);

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings)
        {
            this.audioService = audioService;
            this.dynamicEffectSettings = dynamicEffectSettings;
            this.staticSettings = staticSettings;
        }

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => true;
    }
}
