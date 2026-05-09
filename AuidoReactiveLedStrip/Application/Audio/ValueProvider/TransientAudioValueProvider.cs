using Application.RuntimeSettings;
using Application.Util;

namespace Application.Audio.ValueProvider
{
    public class TransientAudioValueProvider : BaseAudioValueProvider
    {
        private float[]? previousSpectrum;
        private float fastFlux;
        private float fastEnergy;
        private float slowFlux;
        private float slowEnergy;

        protected override void CalculateAudioValue(float maxFrequency)
        {
            StaticSettings? staticSettings = base.staticSettings;
            DynamicEffectSettings? dynamicEffectSettings = base.dynamicSettings;

            if (base.filteredFftData == null ||
                base.minBin == null ||
                base.maxBin == null ||
                staticSettings == null ||
                dynamicEffectSettings == null)
            {
                return;
            }

            if (this.previousSpectrum == null || this.previousSpectrum.Length != base.filteredFftData.Length)
            {
                this.previousSpectrum = new float[base.filteredFftData.Length];
                Array.Copy(base.filteredFftData, this.previousSpectrum, base.filteredFftData.Length);
                return;
            }

            float spectralFlux = 0f;
            float energy = 0f;
            for (int i = 0; i < base.filteredFftData.Length; i++)
            {
                float current = base.filteredFftData[i];
                float previous = this.previousSpectrum[i];
                float delta = current - previous;
                energy += current * current;

                if (delta > 0f)
                {
                    spectralFlux += delta * delta;
                }
            }

            this.fastFlux = MathHelper.Lerp(this.fastFlux, spectralFlux, staticSettings.FastFluxSmoothing);
            this.fastEnergy = MathHelper.Lerp(this.fastEnergy, energy, staticSettings.FastEnergySmoothing);

            this.slowFlux = MathHelper.Lerp(this.slowFlux, spectralFlux, staticSettings.SlowFluxSmoothing);
            this.slowEnergy = MathHelper.Lerp(this.slowEnergy, energy, staticSettings.SlowEnergySmoothing);

            float transientFlux = Math.Max(0f, this.fastFlux - this.slowFlux);
            float transientEnergy = Math.Max(0f, this.fastEnergy - this.slowEnergy);

            float normalizedFlux = transientFlux / Math.Max(this.fastFlux, 0.0001f);
            float normalizedEnergy = transientEnergy / Math.Max(this.fastEnergy, 0.0001f);

            float energyKickPart = Math.Max(1.0f - dynamicEffectSettings.EnergyInfluence, 0.0f) + normalizedEnergy * dynamicEffectSettings.EnergyInfluence;
            float fluxKickPart = Math.Max(1.0f - dynamicEffectSettings.FluxInfluence, 0.0f) + normalizedFlux * dynamicEffectSettings.FluxInfluence;

            float kickValue = fluxKickPart * energyKickPart;
            float result = maxFrequency >= dynamicEffectSettings.MinFreqAmplitude ? kickValue : 0f;

            Array.Copy(base.filteredFftData, this.previousSpectrum, base.filteredFftData.Length);
            base.SetAudioValue(result);
            if (staticSettings.PrintFrequencyInfos)
            {
                Console.WriteLine(
                    $"Kick: {result:F3}\t" +
                    $"TransFlux: {transientFlux:F3}\t" +
                    $"TransEnergy: {transientEnergy:F3}\t" +
                    $"FastFlux: {this.fastFlux:F5}\t" +
                    $"SlowFlux: {this.slowFlux:F5}\t" +
                    $"SlowEnergy: {this.slowEnergy:F3}\t" +
                    $"FastEnergy: {this.fastEnergy:F3}\t");
            }
        }
    }
}