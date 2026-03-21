using Application.Audio.Service;
using Application.Domain;
using Application.Settings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public class AudioPulseEffect : AudioValueBasedEffect
    {
        public AudioPulseEffect(IAudioService audioService, IOptionsMonitor<DynamicSettings> dynamicSettings) 
            : base(audioService, dynamicSettings) {}

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            float value = base.GetCurrentAudioValue();
            
            if (prevStrip != null)
            {
                foreach (var pixel in prevStrip.LedPixels)
                {
                    pixel.Value = value;
                }

                return prevStrip;
            }

            return LedHelper.CreateFilledStrip(length, value);
        }
    }
}
