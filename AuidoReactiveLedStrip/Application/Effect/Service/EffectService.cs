using Application.Domain;
using Application.Settings;
using Microsoft.Extensions.Options;

namespace Application.Effect.Service
{
    public class EffectService : IEffectService
    {
        private readonly IOptionsMonitor<DynamicSettings> dyamicSettings;
        private readonly AudioLineEffekt audioLineEffekt;
        private readonly AudioPulseEffekt audioPulseEffekt;
        private readonly AudioRandomBurstEffect audioRandomBurstEffect;
        private readonly AudioSepctrumEffect audioSepctrumEffect;
        private readonly AudioWaveEffect audioWaveEffect;
        private readonly GpioExternalEffect externalEffect;
        private readonly StaticAscendingValueEffect staticAscendingValueEffect;
        private readonly StaticValueOneEffect staticValueOneEffect;

        private EffectMode? currentEffectMode = null;
        private IEffect? currentEffect = null;
        private LedStrip? prevLedStrip = null;

        public EffectService(
            IOptionsMonitor<DynamicSettings> dyamicSettings,
            AudioLineEffekt audioLineEffekt,
            AudioPulseEffekt audioPulseEffekt,
            AudioRandomBurstEffect audioRandomBurstEffect,
            AudioSepctrumEffect audioSepctrumEffect,
            AudioWaveEffect audioWaveEffect,
            GpioExternalEffect externalEffect,
            StaticAscendingValueEffect staticAscendingValueEffect,
            StaticValueOneEffect staticValueOneEffect
        )
        {
            this.dyamicSettings = dyamicSettings;
            this.audioLineEffekt = audioLineEffekt;
            this.audioPulseEffekt = audioPulseEffekt;
            this.audioRandomBurstEffect = audioRandomBurstEffect;
            this.audioSepctrumEffect = audioSepctrumEffect;
            this.audioWaveEffect = audioWaveEffect;
            this.externalEffect = externalEffect;
            this.staticAscendingValueEffect = staticAscendingValueEffect;
            this.staticValueOneEffect = staticValueOneEffect;
        }

        public void SetEffectMode(EffectMode effectMode)
        {
            if (currentEffectMode == effectMode)
                return;

            if (currentEffect is IStatefulEffect prevStatefulEffect)
                prevStatefulEffect.DisableEffect();

            switch (effectMode)
            {
                case EffectMode.AudioPulsate:
                    currentEffect = audioPulseEffekt;
                    break;
                case EffectMode.AudioLine:
                    currentEffect = audioLineEffekt;
                    break;
                case EffectMode.AudioWave:
                    currentEffect = audioWaveEffect;
                    break;
                case EffectMode.AudioSpectrum:
                    currentEffect = audioSepctrumEffect;
                    break;
                case EffectMode.AudioParticle:
                    currentEffect = audioRandomBurstEffect;
                    break;
                case EffectMode.StaticAscending:
                    currentEffect = staticAscendingValueEffect;
                    break;
                case EffectMode.StaticValueOne:
                    currentEffect = staticValueOneEffect;
                    break;
                case EffectMode.External:
                    currentEffect = externalEffect;
                    break;
            }

            if (currentEffect != null)
            {
                if (currentEffect is IStatefulEffect newStatefulEffect)
                    newStatefulEffect.EnableEffect();

                if (currentEffect.IsStatic)
                {
                    DynamicSettings dynamicSettings = dyamicSettings.CurrentValue;
                    prevLedStrip = currentEffect.RenderEffekt(prevLedStrip, dynamicSettings.LedCount);
                }
            }

            currentEffectMode = effectMode;
        }

        public LedStrip? GetRenderedLedStrip()
        {
            DynamicSettings dynamicSettings = dyamicSettings.CurrentValue;
            if (prevLedStrip != null && dynamicSettings.LedCount != prevLedStrip.LedPixels.Count)
                prevLedStrip = null;

            if (currentEffect == null)
                return null;

            if (!currentEffect.IsStatic || prevLedStrip == null)
                prevLedStrip = currentEffect.RenderEffekt(prevLedStrip, dynamicSettings.LedCount);

            return prevLedStrip;
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
