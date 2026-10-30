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

            float minMag = Math.Max(AudioUtil.DbToMagnitude(dynamicEffectSettings.MinFreqAmplitude), 1e-6f);
            float minPower = minMag * minMag;

            float normalizedFlux = transientFlux / Math.Max(this.fastFlux, minPower);
            float normalizedEnergy = transientEnergy / Math.Max(this.fastEnergy, minPower);

            float energyKickPart = Math.Max(1.0f - dynamicEffectSettings.EnergyInfluence, 0.0f) + normalizedEnergy * dynamicEffectSettings.EnergyInfluence;
            float fluxKickPart = Math.Max(1.0f - dynamicEffectSettings.FluxInfluence, 0.0f) + normalizedFlux * dynamicEffectSettings.FluxInfluence;

            float kickValue = fluxKickPart * energyKickPart;
            float maxFreqDb = AudioUtil.MagnitudeToDb(maxFrequency);
            bool canIncrease = maxFreqDb >= dynamicEffectSettings.MinFreqAmplitude;
            float result = canIncrease ? kickValue : 0f;

            Array.Copy(base.filteredFftData, this.previousSpectrum, base.filteredFftData.Length);
            base.SetAudioValue(result, canIncrease);
            if (staticSettings.PrintFrequencyInfos)
            {
                float fastFluxDb = AudioUtil.PowerToDb(this.fastFlux);
                float slowFluxDb = AudioUtil.PowerToDb(this.slowFlux);
                float deltaFluxDb = fastFluxDb - slowFluxDb;

                float fastEnergyDb = AudioUtil.PowerToDb(this.fastEnergy);
                float slowEnergyDb = AudioUtil.PowerToDb(this.slowEnergy);
                float deltaEnergyDb = fastEnergyDb - slowEnergyDb;

                Console.WriteLine(
                    $"Max freq: {maxFreqDb,7:F1} dB\t" +
                    $"Kick: {result,5:F3}\t" +
                    $"Flux: {fastFluxDb,7:F1} dB (slow: {slowFluxDb,7:F1} dB, /\\ {deltaFluxDb,5:+0.0;-0.0; 0.0} dB)\t" +
                    $"Energy: {fastEnergyDb,7:F1} dB (slow: {slowEnergyDb,7:F1} dB, /\\ {deltaEnergyDb,5:+0.0;-0.0; 0.0} dB)");
            }
        }
    }
}
