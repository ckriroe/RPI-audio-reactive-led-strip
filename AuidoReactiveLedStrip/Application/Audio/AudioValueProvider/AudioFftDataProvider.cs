using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.AudioValueProvider
{
    public class AudioFftDataProvider : IAudioDataProvider
    {
        private BaseAudioDataProviderSettings? settings;
        private int? minBin = null;
        private int? maxBin = null;

        protected volatile float[]? filteredFftData = null;

        public void SetNewFftData(float[] fftData)
        {
            if (this.minBin == null || this.maxBin == null)
                return;

            this.filteredFftData = fftData[this.minBin.Value..this.maxBin.Value];
            this.ProcessFftData();
        }

        public virtual void Initialize(BaseAudioDataProviderSettings settings)
        {
            this.settings = settings;
            double frequencyRangePerBin = this.settings.SampleRate / (double)this.settings.FftSize;
            this.minBin = (int)Math.Round(this.settings.MinFrequency / frequencyRangePerBin);
            this.maxBin = (int)Math.Round(this.settings.MaxFrequency / frequencyRangePerBin) + 1;
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
