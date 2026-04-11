using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioLineEffect : AudioValueBasedEffect
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
                if (i >= start && i <= end)
                {
                    ledPixels[i].Value = value;
                } 
                else
                {
                    ledPixels[i].Value = 0.0f;
                }
            }

            return prevStrip;
        }
    }
}
