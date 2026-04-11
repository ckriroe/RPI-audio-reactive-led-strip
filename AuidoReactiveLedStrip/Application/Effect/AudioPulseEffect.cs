using Application.Domain;
using Application.Util;

namespace Application.Effect
{
    public class AudioPulseEffect : AudioValueBasedEffect
    {
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
