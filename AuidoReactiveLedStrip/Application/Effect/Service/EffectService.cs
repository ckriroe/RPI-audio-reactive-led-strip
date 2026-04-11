using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Effect.Service
{
    public class EffectService : IEffectService
    {
        private readonly AudioLineEffect audioLineEffect;
        private readonly AudioPulseEffect audioPulseEffect;
        private readonly AudioRandomBurstEffect audioRandomBurstEffect;
        private readonly AudioSepctrumEffect audioSepctrumEffect;
        private readonly AudioWaveEffect audioWaveEffect;
        private readonly GpioExternalEffect externalEffect;
        private readonly StaticAscendingValueEffect staticAscendingValueEffect;
        private readonly StaticValueOneEffect staticValueOneEffect;
        private readonly AudioLineDescendingEffect audioLineDescendingEffect;

        private EffectMode? currentEffectMode = null;
        private IEffect? currentEffect = null;
        private LedStrip? prevLedStrip = null;
        private DynamicEffectSettings? dynamicSettings = null;

        public EffectService(
            AudioLineEffect audioLineEffect,
            AudioPulseEffect audioPulseEffect,
            AudioRandomBurstEffect audioRandomBurstEffect,
            AudioSepctrumEffect audioSepctrumEffect,
            AudioWaveEffect audioWaveEffect,
            GpioExternalEffect externalEffect,
            StaticAscendingValueEffect staticAscendingValueEffect,
            StaticValueOneEffect staticValueOneEffect,
            AudioLineDescendingEffect audioLineDescendingEffect
        )
        {
            this.audioLineEffect = audioLineEffect;
            this.audioPulseEffect = audioPulseEffect;
            this.audioRandomBurstEffect = audioRandomBurstEffect;
            this.audioSepctrumEffect = audioSepctrumEffect;
            this.audioWaveEffect = audioWaveEffect;
            this.externalEffect = externalEffect;
            this.staticAscendingValueEffect = staticAscendingValueEffect;
            this.staticValueOneEffect = staticValueOneEffect;
            this.audioLineDescendingEffect = audioLineDescendingEffect;
        }

        public void ApplySettings(IAudioService affectedAudioService, StaticSettings staticSettings, DynamicEffectSettings dynamicSettings)
        {
            this.dynamicSettings = dynamicSettings;
            this.audioLineEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.audioPulseEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.audioRandomBurstEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.audioSepctrumEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.audioWaveEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.externalEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.staticAscendingValueEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.staticValueOneEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);
            this.audioLineDescendingEffect.ApplySettings(affectedAudioService, staticSettings, dynamicSettings);

            this.SetEffectMode(dynamicSettings.EffectMode);
            affectedAudioService.SetAudioMode(this.GetRequiredAudioMode());
        }

        public LedStrip? GetLedStrip()
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            if (dynamicSettings == null)
                return null;

            if (this.prevLedStrip != null && dynamicSettings.CalculatedLedCount != this.prevLedStrip.LedPixels.Length)
                this.prevLedStrip = null;

            if (this.currentEffect == null)
                return null;

            if (!this.currentEffect.IsStatic || this.prevLedStrip == null)
                this.prevLedStrip = this.currentEffect.RenderEffekt(this.prevLedStrip, dynamicSettings.CalculatedLedCount);

            return this.prevLedStrip;
        }

        public void Reset()
        {
            this.prevLedStrip = null;
            this.audioWaveEffect.Reset();
        }

        private AudioServiceMode GetRequiredAudioMode()
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            if (this.currentEffect == null || dynamicSettings == null)
                return AudioServiceMode.None;

            if (this.currentEffect.UseAudioValue)
            {
                if (dynamicSettings.AudioMode == AudioMode.Dynamic)
                    return AudioServiceMode.MovingMax;

                if (dynamicSettings.AudioMode == AudioMode.Static)
                    return AudioServiceMode.Simple;
            }

            if (this.currentEffect.UseAudioFft)
                return AudioServiceMode.Fft;

            return AudioServiceMode.None;
        }

        private void SetEffectMode(EffectMode effectMode)
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            if (this.currentEffectMode == effectMode || dynamicSettings == null)
                return;

            if (this.currentEffect is IStatefulEffect prevStatefulEffect)
                prevStatefulEffect.DisableEffect();

            switch (effectMode)
            {
                case EffectMode.AudioPulsate:
                    this.currentEffect = this.audioPulseEffect;
                    break;
                case EffectMode.AudioLine:
                    this.currentEffect = this.audioLineEffect;
                    break;
                case EffectMode.AudioWave:
                    this.currentEffect = this.audioWaveEffect;
                    break;
                case EffectMode.AudioSpectrum:
                    this.currentEffect = this.audioSepctrumEffect;
                    break;
                case EffectMode.AudioParticle:
                    this.currentEffect = this.audioRandomBurstEffect;
                    break;
                case EffectMode.StaticAscending:
                    this.currentEffect = this.staticAscendingValueEffect;
                    break;
                case EffectMode.StaticValueOne:
                    this.currentEffect = this.staticValueOneEffect;
                    break;
                case EffectMode.External:
                    this.currentEffect = this.externalEffect;
                    break;
                case EffectMode.AudioLineDescending:
                    this.currentEffect = this.audioLineDescendingEffect;
                    break;
            }

            if (this.currentEffect != null)
            {
                if (this.currentEffect is IStatefulEffect newStatefulEffect)
                    newStatefulEffect.EnableEffect();

                if (this.currentEffect.IsStatic)
                {
                    this.prevLedStrip = this.currentEffect.RenderEffekt(this.prevLedStrip, dynamicSettings.CalculatedLedCount);
                }
            }

            this.currentEffectMode = effectMode;
        }
    }
}
