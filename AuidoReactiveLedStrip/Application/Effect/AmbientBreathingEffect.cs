using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AmbientBreathingEffect : IEffect, IStatefulEffect
    {
        private const float TwoPi = MathF.PI * 2.0f;
        private const float SpeedFrequencyDivisor = 100.0f;

        private float phase = 0f;
        private DynamicEffectSettings? dynamicSettings = null;
        private StaticSettings? staticSettings = null;

        public bool IsStatic => false;
        public bool UseAudioFft => false;
        public bool UseAudioValue => false;

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicEffectSettings;
        }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            int framesPerSecond = Math.Max(1, staticSettings.Fps);
            float speed = Math.Max(1, dynamicSettings.Speed);
            float cycleFrequency = speed / SpeedFrequencyDivisor;

            this.phase += (cycleFrequency * TwoPi) / framesPerSecond;
            if (this.phase >= TwoPi)
                this.phase -= TwoPi;

            float normalizedSine = (MathF.Sin(this.phase) + 1.0f) * 0.5f;
            float breathingBrightness = normalizedSine * normalizedSine;

            for (int i = 0; i < length; i++)
            {
                prevStrip.LedPixels[i].Value = breathingBrightness;
            }

            return prevStrip;
        }

        public void EnableEffect()
        {
            this.phase = 0f;
        }

        public void DisableEffect()
        {
            this.phase = 0f;
        }
    }
}
