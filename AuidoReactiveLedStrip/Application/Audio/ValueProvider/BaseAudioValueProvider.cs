using Application.RuntimeSettings;

namespace Application.Audio.ValueProvider
{
    public abstract class BaseAudioValueProvider : AudioFftDataProvider, IAudioValueProvider
    {
        protected volatile float currentValue = 0.0f;

        public float GetAudioValue()
        {
            return this.currentValue;
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
