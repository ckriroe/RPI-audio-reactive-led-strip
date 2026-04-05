using Application.RuntimeSettings;
using Application.Util;

namespace Application.Audio.ValueProvider
{
    public abstract class BaseAudioValueProvider : AudioFftDataProvider, IAudioValueProvider
    {
        private volatile float lastAudioValue = 0.0f;
        private long lastValueRaise = 0;
        private long lastPeakTime = 0;

        public float GetAudioValue()
        {
            return this.lastAudioValue;
        }

        private long lastBeatTime = 0;

        protected void SetAudioValue(float newValue)
        {
            DynamicSettings? dynamicSettings = base.dynamicSettings;
            if (dynamicSettings == null)
                return;

            long now = Environment.TickCount;
            float minMsPerBeat = dynamicSettings.BpmLimit == -1
                ? 0f
                : 1000.0f / (dynamicSettings.BpmLimit / 60.0f);

            newValue = MathF.Pow(newValue, dynamicSettings.AudioResponseCurve);
            bool isAboveThreshold = newValue > dynamicSettings.SaturateThreshold;
            float timeSinceLastBeat = now - lastBeatTime;
            bool onTime = timeSinceLastBeat >= minMsPerBeat;

            if (onTime && isAboveThreshold && newValue > this.lastAudioValue)
            {
                this.lastAudioValue = MathHelper.Lerp(
                    this.lastAudioValue,
                    newValue,
                    dynamicSettings.Saturate
                );

                lastBeatTime = now;
                lastPeakTime = now;
            }
            else
            {
                if (now - lastPeakTime < dynamicSettings.AudioPeakHoldTimeMs)
                    return;

                this.lastAudioValue = MathHelper.Lerp(
                    this.lastAudioValue,
                    0f,
                    1f - dynamicSettings.Fade
                );
            }
        }

        protected sealed override void ProcessFftData()
        {
            if (base.filteredFftData == null)
                return;

            float maxFrequency = base.filteredFftData.Max();
            this.CalculateAudioValue(maxFrequency);
        }

        protected abstract void CalculateAudioValue(float maxFrequency);
    }
}
