namespace Application.Audio.AudioValueProvider
{
    public abstract class BaseAudioValueProvider : AudioFftDataProvider, IAudioValueProvider
    {
        protected BaseAudioValueProviderSettings? settings;
        protected volatile float currentValue = 0.0f;

        public float GetAudioValue()
        {
            return this.currentValue;
        }

        public override void Initialize(BaseAudioDataProviderSettings settings)
        {
            base.Initialize(settings);
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
            Console.WriteLine("Current Value: " + this.currentValue);
        }

        protected abstract void CalculateAudioValue(float maxFrequency);
    }
}
