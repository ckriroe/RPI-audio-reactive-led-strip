using Application.Settings;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.ValueProvider
{
    public class AudioFftDataProvider : IAudioDataProvider
    {
        private int? minBin = null;
        private int? maxBin = null;

        protected volatile float[]? filteredFftData = null;
        protected StaticSettings? staticSettings = null;
        protected DynamicSettings? dynamicSettings = null;

        public void SetNewFftData(float[] fftData)
        {
            if (this.minBin == null || this.maxBin == null)
                return;

            this.filteredFftData = fftData[this.minBin.Value..this.maxBin.Value];
            this.ProcessFftData();
        }

        public void Initialize(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;

            double frequencyRangePerBin = this.staticSettings.SampleRate / (double)this.staticSettings.FftSize;
            int newMinBin = (int)Math.Round(this.dynamicSettings.MinFreq / frequencyRangePerBin);
            int newMaxBin = (int)Math.Round(this.dynamicSettings.MaxFreq / frequencyRangePerBin) + 1;

            if (newMinBin != this.minBin || newMaxBin != this.maxBin)
            {
                this.minBin = newMinBin;
                this.maxBin = newMaxBin;
                Console.WriteLine($"Actual frequency range: {newMinBin * frequencyRangePerBin}hz - {(newMaxBin - 1) * frequencyRangePerBin}hz");
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
    }
}
