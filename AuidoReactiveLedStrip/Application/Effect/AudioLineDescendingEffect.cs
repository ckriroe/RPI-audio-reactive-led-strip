using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioLineDescendingEffect : AudioValueBasedEffect
    {
        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            if (dynamicSettings == null)
                return null;

            float value = base.GetCurrentAudioValue();

            int n = length;
            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            var ledPixels = prevStrip.LedPixels;

            int center = dynamicSettings.EffectOrigin;
            int leftDist = center;
            int rightDist = (n - 1) - center;

            int leftExtent = (int)(leftDist * value);
            int rightExtent = (int)(rightDist * value);

            int start = center - leftExtent;
            int end = center + rightExtent;

            for (int i = 0; i < n; i++)
            {
                float valueToSet = 0.0f;

                if (i >= start && i <= end)
                {
                    if (i == center)
                    {
                        valueToSet = value;
                    } 
                    else if (i < center)
                    {
                        valueToSet = ((i - start) / (float)leftExtent) * value;
                    }
                    else
                    {
                        valueToSet = ((end - i) / (float)rightExtent) * value;
                    }
                }
                
                if (ledPixels[i].Value < valueToSet)
                {
                    ledPixels[i].Value = valueToSet;
                }
                else
                {
                    ledPixels[i].Value *= dynamicSettings.FadeOverTime;
                }
            }

            return prevStrip;
        }
    }
}
