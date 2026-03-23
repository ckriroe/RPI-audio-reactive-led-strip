using Application.Util;
using System.Runtime.Intrinsics.X86;

namespace Application.Audio.ValueProvider
{
    public class SimpleAudioValueProvider : BaseAudioValueProvider
    {
        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (base.dynamicSettings == null || base.staticSettings == null)
                return;

            float frequencyDiff = base.dynamicSettings.MaxFreqAmplitude - base.dynamicSettings.MinFreqAmplitude;
            if (frequencyDiff <= 0.0f)
            {
                base.currentValue = 0.0f;
                return;
            }

            base.currentValue = MathHelper.Clamp((maxFrequency - base.dynamicSettings.MinFreqAmplitude) / frequencyDiff, 0.0f, 1.0f);
            if (base.staticSettings.PrintFrequencyInfos)
                Console.WriteLine($"Max freq: {maxFrequency,15:F5}\tResulting value: {base.currentValue,15:F5}");
        }
    }
}
