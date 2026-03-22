using Application.Settings;

namespace Application.Audio.ValueProvider
{
    public abstract class BaseAudioValueProvider : AudioFftDataProvider, IAudioValueProvider
    {
        protected BaseAudioValueProviderSettings? settings;
        protected volatile float currentValue = 0.0f;

        public float GetAudioValue()
        {
            return this.currentValue;
        }

        public override void Initialize(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            base.Initialize(staticSettings, dynamicSettings);
            if (settings is BaseAudioValueProviderSettings specificSettings)
            {
                this.settings = specificSettings;
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
