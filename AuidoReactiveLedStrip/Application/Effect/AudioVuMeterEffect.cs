using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioVuMeterEffect : AudioValueBasedEffect, IStatefulEffect
    {
        private const float AttackInterpolationFactor = 0.70f;
        private const float DecayScale = 0.25f;
        private const float DecayOffset = 0.05f;
        private const float MinDecay = 0.04f;
        private const float MaxDecay = 0.40f;

        private const float HoldDurationMultiplierMs = 5000f;
        private const float InfiniteHoldThreshold = -0.001f;

        private const float DefaultFramesPerSecond = 60f;
        private const float BaseSpeed = 300f;

        private const float CenterFadeThreshold = 0.08f;
        private const float InactivityCutoffThreshold = 0.005f;
        private const float PeakVisibilityThreshold = 0.001f;

        private float smoothBar = 0f;
        private float smoothPeak = 0f;
        private float peakBrightness = 0f;
        private long peakHoldUntil = 0;

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            float audioValue = MathHelper.Clamp(base.GetCurrentAudioValue(), 0.0f, 1.0f);
            int origin = Math.Clamp(dynamicSettings.EffectOrigin, 0, length - 1);
            int peakSize = Math.Max(1, dynamicSettings.ParticleSize);
            int halfPeak = peakSize / 2;

            int spaceLeft = origin;
            int spaceRight = (length - 1) - origin;

            int maxLeftExtent = Math.Max(0, spaceLeft - halfPeak);
            int maxRightExtent = Math.Max(0, spaceRight - halfPeak);

            if (audioValue >= this.smoothBar)
            {
                this.smoothBar = MathHelper.Lerp(this.smoothBar, audioValue, AttackInterpolationFactor);
            }
            else
            {
                float barDecay = Math.Clamp((1.0f - dynamicSettings.Fade) * DecayScale + DecayOffset, MinDecay, MaxDecay);
                this.smoothBar = MathHelper.Lerp(this.smoothBar, audioValue, barDecay);
            }

            long currentTicksMs = Environment.TickCount;
            float fadeOverTimeSetting = dynamicSettings.FadeOverTime;
            bool isInfiniteHold = fadeOverTimeSetting < InfiniteHoldThreshold;
            float effectiveFadeOverTime = isInfiniteHold ? MathF.Abs(fadeOverTimeSetting) : Math.Max(0.0f, fadeOverTimeSetting);

            int holdDurationMs = isInfiniteHold ? int.MaxValue : (int)(effectiveFadeOverTime * HoldDurationMultiplierMs);

            float framesPerSecond = staticSettings.Fps > 0 ? (float)staticSettings.Fps : DefaultFramesPerSecond;
            float speed = dynamicSettings.Speed > 0 ? (float)dynamicSettings.Speed : BaseSpeed;
            float dropSpeed = (speed / BaseSpeed) / framesPerSecond;

            if (this.smoothBar >= this.smoothPeak)
            {
                this.smoothPeak = this.smoothBar;
                this.peakBrightness = 1.0f;
                this.peakHoldUntil = currentTicksMs + holdDurationMs;
            }
            else if (!isInfiniteHold && currentTicksMs > this.peakHoldUntil)
            {
                this.smoothPeak = Math.Max(this.smoothBar, this.smoothPeak - dropSpeed);
            }

            int leftBarLength = (int)MathF.Round(this.smoothBar * spaceLeft);
            int rightBarLength = (int)MathF.Round(this.smoothBar * spaceRight);

            int leftPeakCenterPosition = (int)MathF.Round(this.smoothPeak * maxLeftExtent);
            int rightPeakCenterPosition = (int)MathF.Round(this.smoothPeak * maxRightExtent);

            float centerFade = Math.Clamp(this.smoothPeak / CenterFadeThreshold, 0.0f, 1.0f);
            float effectivePeakBrightness = this.peakBrightness * centerFade;

            if (this.smoothPeak <= InactivityCutoffThreshold && this.smoothBar <= InactivityCutoffThreshold)
            {
                this.smoothPeak = 0f;
                this.peakBrightness = 0f;
                effectivePeakBrightness = 0f;
            }

            int leftPeakMinDistance = leftPeakCenterPosition - halfPeak;
            int leftPeakMaxDistance = leftPeakMinDistance + peakSize - 1;

            int rightPeakMinDistance = rightPeakCenterPosition - halfPeak;
            int rightPeakMaxDistance = rightPeakMinDistance + peakSize - 1;

            for (int i = 0; i < length; i++)
            {
                float pixelValue = 0.0f;

                if (i <= origin)
                {
                    int distanceFromOrigin = origin - i;

                    if (distanceFromOrigin <= leftBarLength)
                    {
                        pixelValue = audioValue;
                    }

                    if (spaceLeft > 0 && distanceFromOrigin >= leftPeakMinDistance && distanceFromOrigin <= leftPeakMaxDistance && effectivePeakBrightness > PeakVisibilityThreshold)
                    {
                        if (effectivePeakBrightness > pixelValue)
                            pixelValue = effectivePeakBrightness;
                    }
                }
                else
                {
                    int distanceFromOrigin = i - origin;

                    if (distanceFromOrigin <= rightBarLength)
                    {
                        pixelValue = audioValue;
                    }

                    if (spaceRight > 0 && distanceFromOrigin >= rightPeakMinDistance && distanceFromOrigin <= rightPeakMaxDistance && effectivePeakBrightness > PeakVisibilityThreshold)
                    {
                        if (effectivePeakBrightness > pixelValue)
                            pixelValue = effectivePeakBrightness;
                    }
                }

                prevStrip.LedPixels[i].Value = MathHelper.Clamp(pixelValue, 0.0f, 1.0f);
            }

            return prevStrip;
        }

        public void EnableEffect()
        {
            this.Reset();
        }

        public void DisableEffect()
        {
            this.Reset();
        }

        public void Reset()
        {
            this.smoothBar = 0f;
            this.smoothPeak = 0f;
            this.peakBrightness = 0f;
            this.peakHoldUntil = 0;
        }
    }
}
