using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class MeteorEffect : AudioValueBasedEffect, IStatefulEffect
    {
        private const int ForwardDirection = 1;
        private const int BackwardDirection = -1;
        private const int ParticleSizeDivisor = 4;
        private const float MinFade = 0.001f;
        private const float MaxFade = 0.999f;
        private const float MaxBrightness = 1.0f;

        private float headPosition = 0f;
        private int direction = ForwardDirection;

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            int framesPerSecond = Math.Max(1, staticSettings.Fps);
            float speed = Math.Max(1.0f, (float)dynamicSettings.Speed);
            float stepDistance = speed / framesPerSecond;
            bool bouncy = dynamicSettings.BouncyWave;

            float fade = Math.Clamp(dynamicSettings.Fade, MinFade, MaxFade);
            for (int i = 0; i < length; i++)
            {
                prevStrip.LedPixels[i].Value *= fade;
            }

            float startPosition = this.headPosition;
            float endPosition = startPosition + (this.direction * stepDistance);
            int headSize = Math.Max(1, dynamicSettings.ParticleSize / ParticleSizeDivisor);

            if (bouncy)
            {
                if (endPosition >= length - 1)
                {
                    DrawContinuousLine(prevStrip, (int)startPosition, length - 1, length);
                    float overshoot = endPosition - (length - 1);
                    this.direction = BackwardDirection;
                    startPosition = length - 1;
                    endPosition = Math.Max(0f, (length - 1) - overshoot);
                }
                else if (endPosition <= 0)
                {
                    DrawContinuousLine(prevStrip, (int)startPosition, 0, length);
                    float undershoot = -endPosition;
                    this.direction = ForwardDirection;
                    startPosition = 0;
                    endPosition = Math.Min(length - 1f, undershoot);
                }
            }
            else
            {
                if (endPosition >= length)
                {
                    DrawContinuousLine(prevStrip, (int)startPosition, length - 1, length);
                    startPosition = 0;
                    endPosition %= length;
                }
                else if (endPosition < 0)
                {
                    DrawContinuousLine(prevStrip, (int)startPosition, 0, length);
                    startPosition = length - 1;
                    endPosition = (endPosition % length + length) % length;
                }
            }

            DrawContinuousLine(prevStrip, (int)startPosition, (int)endPosition, length);
            this.headPosition = endPosition;

            int headIndex = (int)this.headPosition;
            for (int headOffset = 0; headOffset < headSize; headOffset++)
            {
                int pixelIndex = headIndex - (this.direction * headOffset);
                if (pixelIndex >= 0 && pixelIndex < length)
                {
                    prevStrip.LedPixels[pixelIndex].Value = MaxBrightness;
                }
            }

            return prevStrip;
        }

        private static void DrawContinuousLine(LedStrip strip, int startIndex, int endIndex, int length)
        {
            int minIndex = Math.Clamp(Math.Min(startIndex, endIndex), 0, length - 1);
            int maxIndex = Math.Clamp(Math.Max(startIndex, endIndex), 0, length - 1);
            for (int i = minIndex; i <= maxIndex; i++)
            {
                strip.LedPixels[i].Value = MaxBrightness;
            }
        }

        public void EnableEffect()
        {
            this.headPosition = 0f;
            this.direction = ForwardDirection;
        }

        public void DisableEffect()
        {
            this.headPosition = 0f;
            this.direction = ForwardDirection;
        }
    }
}
