using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public abstract class AudioValueBasedEffect : IEffect
    {
        private readonly IAudioService audioService;
        protected readonly IOptionsMonitor<DynamicSettings> dynamicSettings;

        private float lastAudioValue;

        protected AudioValueBasedEffect(IAudioService audioService, IOptionsMonitor<DynamicSettings> dynamicSettings)
        {
            this.audioService = audioService;
            this.dynamicSettings = dynamicSettings;
        }

        protected float GetCurrentAudioValue()
        {
            DynamicSettings dynamicSettings = this.dynamicSettings.CurrentValue;
            float newAudioValue = this.audioService.GetCurrentAudioValue() ?? 0.0f;
            if (newAudioValue > this.lastAudioValue && newAudioValue > dynamicSettings.SaturateThreshold)
                this.lastAudioValue = MathHelper.Lerp(this.lastAudioValue, newAudioValue, dynamicSettings.Saturate);
            else
                this.lastAudioValue *= dynamicSettings.Fade;

            return this.lastAudioValue * dynamicSettings.ValueIncreaseFactor;
        }

        public abstract LedStrip? RenderEffekt(LedStrip? prevStrip, int length);

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => true;
    }
}
