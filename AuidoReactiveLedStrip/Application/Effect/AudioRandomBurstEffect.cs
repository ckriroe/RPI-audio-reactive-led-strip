using Application.Audio.Service;
using Application.Domain;
using Application.Settings;
using Application.Util;
using Microsoft.Extensions.Options;
using System.Drawing;

namespace Application.Effect
{
    public class AudioRandomBurstEffect : AudioValueBasedEffect
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;

        public AudioRandomBurstEffect(IAudioService audioService, IOptionsMonitor<DynamicSettings> dynamicSettings, IOptionsMonitor<StaticSettings> staticSettings)
            : base(audioService, dynamicSettings)
        {
            this.staticSettings = staticSettings;
        }

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            var dynamicSettings = base.dynamicSettings.CurrentValue;
            var staticSettings = this.staticSettings.CurrentValue;

            float value = base.GetCurrentAudioValue();

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            var prevPixels = prevStrip.LedPixels;

            for (int i = 0; i < length; i++)
            {
                prevPixels[i].Value = prevPixels[i].Value * dynamicSettings.Fade;
            }

            int intensityRadius = (int)dynamicSettings.ValueIncreaseFactor;

            float probabilityFactor = (dynamicSettings.Speed / (float)staticSettings.MaxEffectSpeed) * 0.01f;
            float effectiveProb = Math.Min(1f, value * probabilityFactor);

            for (int i = 0; i < length; i++)
            {
                if (Random.Shared.NextDouble() < effectiveProb)
                {
                    prevPixels[i] = new LedPixel(value);

                    for (int offset = 1; offset <= intensityRadius; offset++)
                    {
                        foreach (int neighbor in new[] { i - offset, i + offset })
                        {
                            if (neighbor >= 0 && neighbor < length)
                            {
                                float t = (float)offset / (intensityRadius + 1);
                                float interpVal = value * (1f - t);

                                prevPixels[neighbor] = new LedPixel(interpVal);
                            }
                        }
                    }
                }
            }

            return prevStrip;
        }
    }
}
