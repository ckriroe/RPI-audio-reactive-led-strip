using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Effect
{
    public interface IEffect
    {
        bool IsStatic { get; }

        bool UseAudioFft { get; }

        bool UseAudioValue { get; }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length);

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings);
    }
}
