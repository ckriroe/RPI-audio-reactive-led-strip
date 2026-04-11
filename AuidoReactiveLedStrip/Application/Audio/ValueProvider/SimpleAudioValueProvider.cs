using Application.Util;

namespace Application.Audio.ValueProvider
{
    public class SimpleAudioValueProvider : BaseAudioValueProvider
    {
        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (base.dynamicSettings == null || base.staticSettings == null)
                return;

            float newAudioValue = 0.0f;
            float frequencyDiff = base.dynamicSettings.MaxFreqAmplitude - base.dynamicSettings.MinFreqAmplitude;
            if (frequencyDiff > 0.0f)
            {
                newAudioValue = MathHelper.Clamp((maxFrequency - base.dynamicSettings.MinFreqAmplitude) / frequencyDiff, 0.0f, 1.0f);
            }

            base.SetAudioValue(newAudioValue);
            if (base.staticSettings.PrintFrequencyInfos)
                Console.WriteLine($"Max freq: {maxFrequency,15:F5}\tResulting value: {newAudioValue,15:F5}");
        }
    }
}
