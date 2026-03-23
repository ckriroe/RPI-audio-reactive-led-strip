using Application.Settings;

namespace Application.Audio.ValueProvider
{
    public class AudioFftDataProvider : IAudioDataProvider
    {
        private float? frequencyRangePerBin = null;
        private int? minBin = null;
        private int? maxBin = null;
        private bool isActive;

        protected volatile float[]? filteredFftData = null;
        protected StaticSettings? staticSettings = null;
        protected DynamicSettings? dynamicSettings = null;

        public void SetNewFftData(float[] fftData)
        {
            if (this.minBin == null || this.maxBin == null || !this.isActive)
                return;

            this.filteredFftData = fftData[this.minBin.Value..this.maxBin.Value];
            this.ProcessFftData();
        }

        public void Initialize(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;

            this.frequencyRangePerBin = this.staticSettings.SampleRate / (float)this.staticSettings.FftSize;
            int newMinBin = (int)Math.Round(this.dynamicSettings.MinFreq / this.frequencyRangePerBin.Value);
            int newMaxBin = (int)Math.Round(this.dynamicSettings.MaxFreq / this.frequencyRangePerBin.Value) + 1;

            if (newMinBin != this.minBin || newMaxBin != this.maxBin)
            {
                this.minBin = newMinBin;
                this.maxBin = newMaxBin;
                this.PrintFrequencyInfo();
            }            
        }

        public float[]? GetCurrentFftData()
        {
            return this.filteredFftData;
        }

        protected virtual void ProcessFftData()
        {
            // Do nothing by default
        }

        public void SetActive(bool isActive)
        {
            this.isActive = isActive;
            this.PrintFrequencyInfo();
        }

        private void PrintFrequencyInfo()
        {
            if (this.isActive)
            {
                Console.WriteLine($"Actual frequency range for {this.GetType().Name}: {this.minBin * this.frequencyRangePerBin}hz - {(this.maxBin - 1) * this.frequencyRangePerBin}hz");
            }
        }
    }
}
