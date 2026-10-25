using Application.Util;

namespace Application.Audio.ValueProvider
{
    public class MovingMaxAudioValueProvider : BaseAudioValueProvider
    {
        private LimitedBuffer<float>? lastExtraOrdanarySampleBuffer = null;
        private float lastAproxMaxFrequency = -1.0f;
        private int lastAproxMaxFreqEval = -1;

        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (base.staticSettings == null || base.dynamicSettings == null)
                return;

            float maxFreqDb = AudioUtil.MagnitudeToDb(maxFrequency);
            float minMag = AudioUtil.DbToMagnitude(base.dynamicSettings.MinFreqAmplitude);
            float maxMag = AudioUtil.DbToMagnitude(base.dynamicSettings.MaxFreqAmplitude);

            if (maxFreqDb > this.dynamicSettings.MaxFreqAmplitude)
            {
                maxFrequency = maxMag;
                maxFreqDb = this.dynamicSettings.MaxFreqAmplitude;
            }

            bool canIncrease = maxFreqDb >= this.dynamicSettings.MinFreqAmplitude;

            char movementIdentifier = '-';
            float maxFreqMultiplier = base.staticSettings.MaxFreqAmplitudeValueMultiplier;
            if (canIncrease && maxFrequency * maxFreqMultiplier > this.lastAproxMaxFrequency)
            {
                this.lastAproxMaxFrequency = Math.Max(this.lastAproxMaxFrequency, MathHelper.Lerp(this.lastAproxMaxFrequency, maxFrequency * maxFreqMultiplier, base.staticSettings.MaxFreqAmplitudeIncreaseRatio));
                this.lastAproxMaxFreqEval = Environment.TickCount;
                movementIdentifier = '▲';
            } 
            else if (maxFrequency * maxFreqMultiplier > this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * base.staticSettings.MaxFreqAmplitudeProlongerThreshholdPercent)
            {
                this.lastAproxMaxFrequency = Math.Min(this.lastAproxMaxFrequency, MathHelper.Lerp(this.lastAproxMaxFrequency, maxFrequency * maxFreqMultiplier, base.staticSettings.MaxFreqAmplitudeDecreaseRatio));
                this.lastAproxMaxFreqEval = Environment.TickCount;
                maxFrequency = this.lastAproxMaxFrequency;
                movementIdentifier = '▼';
            } 
            else if (Environment.TickCount - this.lastAproxMaxFreqEval > this.staticSettings.MaxFreqAmplitudeTTL)
            {
                this.lastAproxMaxFrequency *= (1.0f - this.staticSettings.MaxFreqAmplitudeDecayRate);
            }

            if (this.lastExtraOrdanarySampleBuffer == null || this.lastExtraOrdanarySampleBuffer.MaxSize != this.dynamicSettings.MeanValueBufferSize)
                this.lastExtraOrdanarySampleBuffer = new LimitedBuffer<float>(this.dynamicSettings.MeanValueBufferSize);

            if (maxFrequency < this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * this.dynamicSettings.MeanValueThreshold)
                this.lastExtraOrdanarySampleBuffer.Add(maxFrequency);

            float avg = this.lastExtraOrdanarySampleBuffer.Items.Any() ? this.lastExtraOrdanarySampleBuffer.Items.Average() : 0.0f;

            float adjustedFreqValue;
            if (canIncrease)
            {
                adjustedFreqValue = maxFrequency;
            } 
            else
            {
                float scaleFactor = this.staticSettings.BelowMinFreqAmplitudeFunctionFactor;
                adjustedFreqValue = minMag -
                    (1.0f / scaleFactor) +
                    (1.0f / scaleFactor) *
                    (float)Math.Exp(scaleFactor * (maxFrequency - minMag));

                adjustedFreqValue = Math.Max(0.0f, adjustedFreqValue);
            }

            float newAudioValue = 0.0f;
            if (this.lastAproxMaxFrequency >= minMag && this.lastAproxMaxFrequency > 0.0f)
                newAudioValue = Math.Max(0, (adjustedFreqValue - avg) / this.lastAproxMaxFrequency);

            base.SetAudioValue(newAudioValue, canIncrease);
            if (base.staticSettings.PrintFrequencyInfos)
                Console.WriteLine($"Last approx. max freq: {AudioUtil.MagnitudeToDb(this.lastAproxMaxFrequency),7:F1} dB\t\tCurrrent avg. mean amplitude: {AudioUtil.MagnitudeToDb(avg),7:F1} dB\t\tMax freq: {maxFreqDb,7:F1} dB\t\tResulting value: {newAudioValue,15:F5}\t{movementIdentifier}");
        }
    }
}
