using Application.Domain;
using Application.RuntimeSettings;
using Microsoft.Extensions.Options;

namespace Application.Effect.Service
{
    public class EffectService : IEffectService
    {
        private readonly IOptionsMonitor<DynamicSettings> dyamicSettings;
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

        public EffectService(
            IOptionsMonitor<DynamicSettings> dyamicSettings,
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
            this.dyamicSettings = dyamicSettings;
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

        public void SetEffectMode(EffectMode effectMode)
        {
            if (this.currentEffectMode == effectMode)
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
                    DynamicSettings dynamicSettings = this.dyamicSettings.CurrentValue;
                    this.prevLedStrip = this.currentEffect.RenderEffekt(this.prevLedStrip, dynamicSettings.LedCount);
                }
            }

            currentEffectMode = effectMode;
        }

        public LedStrip? GetRenderedLedStrip()
        {
            DynamicSettings dynamicSettings = this.dyamicSettings.CurrentValue;
            if (this.prevLedStrip != null && dynamicSettings.LedCount != this.prevLedStrip.LedPixels.Length)
                this.prevLedStrip = null;

            if (this.currentEffect == null)
                return null;

            if (!this.currentEffect.IsStatic || this.prevLedStrip == null)
                this.prevLedStrip = this.currentEffect.RenderEffekt(this.prevLedStrip, dynamicSettings.LedCount);

            return this.prevLedStrip;
        }

        public AudioServiceMode GetRequiredAudioMode()
        {
            if (this.currentEffect == null)
                return AudioServiceMode.None;

            if (this.currentEffect.UseAudioValue)
            {
                DynamicSettings dynamicSettings = this.dyamicSettings.CurrentValue;
                if (dynamicSettings.AudioMode == AudioMode.Dynamic)
                    return AudioServiceMode.MovingMax;

                if (dynamicSettings.AudioMode == AudioMode.Static)
                    return AudioServiceMode.Simple;
            }

            if (this.currentEffect.UseAudioFft)
                return AudioServiceMode.Fft;

            return AudioServiceMode.None;
        }
    }
}
