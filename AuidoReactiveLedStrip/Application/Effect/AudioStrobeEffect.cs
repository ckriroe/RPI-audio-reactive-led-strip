using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioStrobeEffect : AudioValueBasedEffect, IStatefulEffect
    {
        private const float SpeedToRateDivisor = 15.0f;
        private const float MinStrobeRateHz = 2.0f;
        private const float MaxStrobeFpsMultiplier = 0.5f;
        private const float AudioThreshold = 0.02f;

        private float phase = 0f;
        private float currentFlashValue = 0f;

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            float audioValue = base.GetCurrentAudioValue();
            prevStrip.CurrentAudioValue = audioValue;
            int framesPerSecond = Math.Max(1, staticSettings.Fps);

            float maxStrobeRate = Math.Clamp(dynamicSettings.Speed / SpeedToRateDivisor, MinStrobeRateHz, framesPerSecond * MaxStrobeFpsMultiplier);
            float strobeRate = audioValue * maxStrobeRate;

            if (audioValue > AudioThreshold)
            {
                this.phase += strobeRate / framesPerSecond;
                if (this.phase >= 1.0f)
                {
                    this.phase -= 1.0f;
                    this.currentFlashValue = 1.0f;
                }
                else
                {
                    this.currentFlashValue *= dynamicSettings.Fade;
                }
            }
            else
            {
                this.phase = 0f;
                this.currentFlashValue *= dynamicSettings.Fade;
            }

            float flashBrightness = MathHelper.Clamp(this.currentFlashValue, 0.0f, 1.0f);
            for (int i = 0; i < length; i++)
            {
                prevStrip.LedPixels[i].Value = flashBrightness;
            }

            return prevStrip;
        }

        public void EnableEffect()
        {
            this.phase = 0f;
            this.currentFlashValue = 0f;
        }

        public void DisableEffect()
        {
            this.phase = 0f;
            this.currentFlashValue = 0f;
        }
    }
}
