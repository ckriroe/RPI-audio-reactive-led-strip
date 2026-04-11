using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioRandomBurstEffect : AudioValueBasedEffect
    {
        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            float value = base.GetCurrentAudioValue();

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            var prevPixels = prevStrip.LedPixels;

            for (int i = 0; i < length; i++)
            {
                prevPixels[i].Value = prevPixels[i].Value * dynamicSettings.Fade;
            }

            int intensityRadius = dynamicSettings.ParticleSize;

            float probabilityFactor = (dynamicSettings.Speed / (float)staticSettings.MaxEffectSpeed) * 0.01f;
            float effectiveProb = Math.Min(1f, value * probabilityFactor);

            for (int i = 0; i < length; i++)
            {
                if (Random.Shared.NextDouble() < effectiveProb)
                {
                    prevPixels[i].Value = value;

                    for (int offset = 1; offset <= intensityRadius; offset++)
                    {
                        foreach (int neighbor in new[] { i - offset, i + offset })
                        {
                            if (neighbor >= 0 && neighbor < length)
                            {
                                float t = (float)offset / (intensityRadius + 1);
                                float interpVal = value * (1f - t);

                                if (prevPixels[neighbor].Value < interpVal)
                                    prevPixels[neighbor].Value = interpVal;
                            }
                        }
                    }
                }
            }

            return prevStrip;
        }
    }
}
