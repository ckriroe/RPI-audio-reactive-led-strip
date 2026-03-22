using Application.Settings;
using Application.Util;
using System.Transactions;

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

            if (maxFrequency > this.lastAproxMaxFrequency)
            {
                this.lastAproxMaxFrequency = MathHelper.Lerp(this.lastAproxMaxFrequency, maxFrequency, base.staticSettings.MaxFreqAmplitudeIncreaseRatio);
                this.lastAproxMaxFreqEval = Environment.TickCount;
            } 
            else if (maxFrequency > this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * base.staticSettings.MaxFreqAmplitudeProlongerThreshholdPercent)
            {
                this.lastAproxMaxFrequency = MathHelper.Lerp(this.lastAproxMaxFrequency, maxFrequency, base.staticSettings.MaxFreqAmplitudeDecreaseRatio);
                this.lastAproxMaxFreqEval = Environment.TickCount;
                maxFrequency = this.lastAproxMaxFrequency;
            } 
            else if (Environment.TickCount - this.lastAproxMaxFreqEval > this.staticSettings.MaxFreqAmplitudeTTL)
            {
                this.lastAproxMaxFrequency *= (1.0f - this.staticSettings.MaxFreqAmplitudeDecayRate);
            }

            if (this.lastExtraOrdanarySampleBuffer == null || this.lastExtraOrdanarySampleBuffer.MaxSize != this.dynamicSettings.MeanValueBufferSize)
                this.lastExtraOrdanarySampleBuffer = new LimitedBuffer<float>(this.dynamicSettings.MeanValueBufferSize);

            if (maxFrequency < this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * this.dynamicSettings.MeanValueThreshold || !this.lastExtraOrdanarySampleBuffer.Items.Any())
                this.lastExtraOrdanarySampleBuffer.Add(maxFrequency);

            float avg = this.lastExtraOrdanarySampleBuffer.Items.Average();

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

            base.currentValue = Math.Max(0, (adjustedFreqValue - avg) / this.lastAproxMaxFrequency);
        }
    }
}
