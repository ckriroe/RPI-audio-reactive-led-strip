using Application.Util;

namespace Application.Audio.ValueProvider
{
    public class SimpleAudioValueProvider : BaseAudioValueProvider
    {
        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (base.dynamicSettings == null || base.staticSettings == null)
                return;

            float maxFreqDb = AudioUtil.MagnitudeToDb(maxFrequency);
            float newAudioValue = 0.0f;
            float frequencyDiff = base.dynamicSettings.MaxFreqAmplitude - base.dynamicSettings.MinFreqAmplitude;
            if (frequencyDiff > 0.0f)
            {
                newAudioValue = MathHelper.Clamp((maxFreqDb - base.dynamicSettings.MinFreqAmplitude) / frequencyDiff, 0.0f, 1.0f);
            }

            bool canIncrease = maxFreqDb >= base.dynamicSettings.MinFreqAmplitude;
            base.SetAudioValue(newAudioValue, canIncrease);

            if (base.staticSettings.PrintFrequencyInfos)
                Console.WriteLine($"Max freq: {maxFreqDb,7:F1} dB\tResulting value: {newAudioValue,15:F5}");
        }
    }
}
