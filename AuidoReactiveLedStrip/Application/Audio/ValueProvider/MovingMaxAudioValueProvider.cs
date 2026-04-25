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

            if (maxFrequency > this.dynamicSettings.MaxFreqAmplitude)
                maxFrequency = this.dynamicSettings.MaxFreqAmplitude;

            char movementIdentifier = '-';
            float maxFreqMultiplier = base.staticSettings.MaxFreqAmplitudeValueMultiplier;
            if (maxFrequency * maxFreqMultiplier > this.lastAproxMaxFrequency)
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
            if (maxFrequency > this.dynamicSettings.MinFreqAmplitude)
            {
                adjustedFreqValue = maxFrequency;
            } 
            else
            {
                float scaleFactor = this.staticSettings.BelowMinFreqAmplitudeFunctionFactor;
                adjustedFreqValue = this.dynamicSettings.MinFreqAmplitude -
                    (1.0f / scaleFactor) +
                    (1.0f / scaleFactor) *
                    (float)Math.Exp(scaleFactor * (maxFrequency - this.dynamicSettings.MinFreqAmplitude));

                adjustedFreqValue = Math.Max(0.0f, adjustedFreqValue);
            }

            float newAudioValue = Math.Max(0, (adjustedFreqValue - avg) / this.lastAproxMaxFrequency);
            base.SetAudioValue(newAudioValue);

            if (base.staticSettings.PrintFrequencyInfos)
                Console.WriteLine($"Last approx. max freq: {this.lastAproxMaxFrequency,15:F5}\t\tCurrrent avg. mean amplitude: {avg,15:F5}\t\tMax freq: {maxFrequency,15:F5}\t\tResulting value: {newAudioValue,15:F5}\t{movementIdentifier}");
        }
    }
}
