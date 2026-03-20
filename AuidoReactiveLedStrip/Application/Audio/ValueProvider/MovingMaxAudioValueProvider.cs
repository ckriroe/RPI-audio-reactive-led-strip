using Application.Util;
using System.Transactions;

namespace Application.Audio.ValueProvider
{
    public class MovingMaxAudioValueProvider : BaseAudioValueProvider
    {
        private MovingMaxAudioValueProviderSettings? sepcificSettings = null;


        private LimitedBuffer<float>? lastExtraOrdanarySampleBuffer = null;
        private float lastAproxMaxFrequency = -1.0f;
        private int lastAproxMaxFreqEval = -1;

        public override void Initialize(BaseAudioDataProviderSettings settings)
        {
            base.Initialize(settings);
            if (settings is MovingMaxAudioValueProviderSettings sepcificSettings)
            {
                this.sepcificSettings = sepcificSettings;
            }
        }

        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (this.sepcificSettings == null)
                return;

            if (maxFrequency > this.sepcificSettings.MaxFrequencyAmplitude)
                maxFrequency = this.sepcificSettings.MaxFrequencyAmplitude;

            if (maxFrequency > this.lastAproxMaxFrequency)
            {
                this.lastAproxMaxFrequency = (this.lastAproxMaxFrequency + maxFrequency * (this.sepcificSettings.MaxFreqAmplitudeIncreaseRatio - 1)) / this.sepcificSettings.MaxFreqAmplitudeIncreaseRatio;
                this.lastAproxMaxFreqEval = Environment.TickCount;
            } 
            else if (maxFrequency > this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * this.sepcificSettings.MaxFreqAmplitudeProlongerThreshholdPercent)
            {
                this.lastAproxMaxFrequency = (lastAproxMaxFrequency * (this.sepcificSettings.MaxFreqAmplitudeDecreaseRatio - 1) + maxFrequency) / this.sepcificSettings.MaxFreqAmplitudeDecreaseRatio;
                this.lastAproxMaxFreqEval = Environment.TickCount;
                maxFrequency = this.lastAproxMaxFrequency;
            } 
            else if (Environment.TickCount - this.lastAproxMaxFreqEval > this.sepcificSettings.MaxFreqAmplitudeTTL)
            {
                this.lastAproxMaxFrequency *= (1.0f - this.sepcificSettings.MaxFreqAmplitudeDecayRate);
            }

            if (this.lastExtraOrdanarySampleBuffer == null || this.lastExtraOrdanarySampleBuffer.MaxSize != this.sepcificSettings.LastExtraOrdanarySampleBufferSize)
                this.lastExtraOrdanarySampleBuffer = new LimitedBuffer<float>(this.sepcificSettings.LastExtraOrdanarySampleBufferSize);

            if (maxFrequency < this.lastAproxMaxFrequency - this.lastAproxMaxFrequency * this.sepcificSettings.PercentDiffFromMaxToBeExtraOrdanary || !this.lastExtraOrdanarySampleBuffer.Items.Any())
                this.lastExtraOrdanarySampleBuffer.Add(maxFrequency);

            float avg = this.lastExtraOrdanarySampleBuffer.Items.Average();

            float adjustedFreqValue;
            if (maxFrequency > this.sepcificSettings.MinFrequencyAmplitude)
            {
                adjustedFreqValue = maxFrequency;
            } 
            else
            {
                float scaleFactor = this.sepcificSettings.BelowMinFreqAmplitudeFunctionFactor;
                adjustedFreqValue = this.sepcificSettings.MinFrequencyAmplitude -
                    (1.0f / scaleFactor) +
                    (1.0f / scaleFactor) *
                    (float)Math.Exp(scaleFactor * (maxFrequency - this.sepcificSettings.MinFrequencyAmplitude));

                adjustedFreqValue = Math.Max(0.0f, adjustedFreqValue);
            }

            base.currentValue = Math.Max(0, (adjustedFreqValue - avg) / this.lastAproxMaxFrequency);
        }
    }
}
