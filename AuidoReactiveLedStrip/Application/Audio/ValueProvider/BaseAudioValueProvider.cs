using Application.RuntimeSettings;
using Application.Util;

namespace Application.Audio.ValueProvider
{
    public abstract class BaseAudioValueProvider : AudioFftDataProvider, IAudioValueProvider
    {
        private volatile float lastAudioValue = 0.0f;
        private long lastValueRaise = 0;

        public float GetAudioValue()
        {
            return this.lastAudioValue;
        }

        protected void SetAudioValue(float newValue)
        {
            DynamicSettings? dynamicSettings = base.dynamicSettings;
            if (dynamicSettings == null)
                return;

            float minMsPerBeat = 1000.0f / (dynamicSettings.BpmLimit / 60.0f);
            long now = Environment.TickCount;
            bool shouldAllowBeat = dynamicSettings.BpmLimit == -1 || now - this.lastValueRaise > minMsPerBeat;

            if (newValue > this.lastAudioValue && newValue > dynamicSettings.SaturateThreshold && shouldAllowBeat)
            {
                this.lastAudioValue = MathHelper.Lerp(this.lastAudioValue, newValue, dynamicSettings.Saturate);
                this.lastValueRaise = now;
            }
            else
            {
                this.lastAudioValue *= dynamicSettings.Fade;
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
