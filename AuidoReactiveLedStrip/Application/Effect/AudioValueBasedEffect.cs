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

        protected AudioValueBasedEffect(IAudioService audioService, IOptionsMonitor<DynamicSettings> dynamicSettings)
        {
            this.audioService = audioService;
            this.dynamicSettings = dynamicSettings;
        }

        protected float GetCurrentAudioValue()
        {
            DynamicSettings dynamicSettings = this.dynamicSettings.CurrentValue;
            return (this.audioService.GetCurrentAudioValue() ?? 0.0f) * dynamicSettings.ValueIncreaseFactor;
        }

        public abstract LedStrip? RenderEffekt(LedStrip? prevStrip, int length);

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => true;
    }
}
