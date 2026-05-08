using Application.RuntimeSettings;
using Application.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace Application.Audio.ValueProvider
{
    public class BeatDetectionValueProvider : BaseAudioValueProvider
    {
        private readonly Queue<float> _fluxHistory = new();
        private readonly Queue<float> _energyHistory = new();

        private float[]? previousSpectrum;

        private float smoothedFlux;
        private float smoothedEnergy;

        private float peakFlux;
        private float peakEnergy;

        private float outputValue;

        public KickBeatDetectorSettings Settings { get; }

        public BeatDetectionValueProvider()
        {
            Settings = new KickBeatDetectorSettings();
        }

        protected override void CalculateAudioValue(float maxFrequency)
        {
            StaticSettings? staticSettings = base.staticSettings;
            DynamicEffectSettings? dynamicEffectSettings = base.dynamicSettings;

            if (base.wholeFftData == null ||
                base.minBin == null ||
                base.maxBin == null ||
                base.frequencyRangePerBin == null ||
                staticSettings == null ||
                dynamicEffectSettings == null)
            {
                return;
            }

            if (this.previousSpectrum == null)
            {
                this.previousSpectrum = new float[base.wholeFftData.Length];
                Array.Copy(base.wholeFftData, this.previousSpectrum, base.wholeFftData.Length);
                return;
            }

            float spectralFlux = 0f;

            for (int i = base.minBin.Value; i <= base.maxBin.Value; i++)
            {
                float current = base.wholeFftData[i];
                float previous = this.previousSpectrum[i];

                float delta = current - previous;

                // Only positive changes matter
                // We only care about attacks/transients
                if (delta > 0f)
                {
                    // Squaring emphasizes stronger transients
                    spectralFlux += delta * delta;
                }
            }

            maxFrequency = Math.Min(dynamicEffectSettings.MaxFreqAmplitude, maxFrequency);
            this.smoothedFlux = MathHelper.Lerp(this.smoothedFlux, spectralFlux, Settings.FluxSmoothing);
            this.smoothedEnergy = MathHelper.Lerp(this.smoothedEnergy, maxFrequency, Settings.EnergySmoothing);

            //--------------------------------------------
            // 4. Update adaptive histories
            //--------------------------------------------

            PushHistory(_fluxHistory, this.smoothedFlux, Settings.HistorySize);
            PushHistory(_energyHistory, this.smoothedEnergy, Settings.HistorySize);

            float avgFlux = _fluxHistory.Count > 0 ? _fluxHistory.Average() : 0f;
            float avgEnergy = _energyHistory.Count > 0 ? _energyHistory.Average() : 0f;

            //--------------------------------------------
            // 5. Normalize against moving averages
            //--------------------------------------------

            float normalizedFlux = smoothedFlux / Math.Max(avgFlux * Settings.FluxThresholdMultiplier, 0.0001f);
            float normalizedEnergy = smoothedEnergy / Math.Max(avgEnergy * Settings.EnergyThresholdMultiplier, 0.0001f);

            //--------------------------------------------
            // 6. Compute transient sharpness
            //--------------------------------------------

            float sharpness = smoothedFlux / Math.Max(smoothedEnergy, 0.0001f);
            float normalizedSharpness = Math.Clamp(sharpness / Settings.SharpnessNormalization, 0f, 1f);

            //--------------------------------------------
            // 7. Adaptive peak normalization
            //--------------------------------------------

            float decay = 1.0f - staticSettings.MaxFreqAmplitudeDecayRate;
            this.peakFlux = Math.Max(this.peakFlux * decay, normalizedFlux);
            this.peakEnergy = Math.Max(this.peakEnergy * decay, normalizedEnergy);

            normalizedFlux /= Math.Max(this.peakFlux, 0.0001f);
            normalizedEnergy /= Math.Max(this.peakEnergy, 0.0001f);

            //--------------------------------------------
            // 8. Build final kick value
            //--------------------------------------------

            float kickValue = normalizedFlux * normalizedSharpness * (0.5f + normalizedEnergy * 0.5f);

            //--------------------------------------------
            // 10. Attack / release smoothing
            //--------------------------------------------

            float smoothing = kickValue > outputValue ? Settings.AttackSmoothing : Settings.ReleaseSmoothing;
            this.outputValue = Math.Clamp(MathHelper.Lerp(this.outputValue, kickValue, smoothing), 0f, 1f);

            //--------------------------------------------
            // 12. Save spectrum
            //--------------------------------------------

            Array.Copy(base.wholeFftData, previousSpectrum, base.wholeFftData.Length);
            float result = dynamicEffectSettings.MinFreqAmplitude < maxFrequency ? this.outputValue : 0.0f;
            base.SetAudioValue(result);

            if (staticSettings.PrintFrequencyInfos)
            {
                Console.WriteLine(
                    $"Kick: {result:F3}\t" +
                    $"Flux: {smoothedFlux:F5}\t" +
                    $"Energy: {smoothedEnergy:F0}\t" +
                    $"Sharpness: {sharpness:F3}\t" +
                    $"NormFlux: {normalizedFlux:F3}\t" +
                    $"NormEnergy: {normalizedEnergy:F3}\t" +
                    $"AvgFlux: {avgFlux:F0}\t" +
                    $"AvgEnergy: {avgEnergy:F0}\t" +
                    $"FluxDiff: {Math.Max(0f, smoothedFlux - avgFlux):F0}\t" +
                    $"EnergyDiff: {Math.Max(0f, smoothedEnergy - avgEnergy):F0}\t");
            }
        }

        private static void PushHistory(
            Queue<float> queue,
            float value,
            int max)
        {
            queue.Enqueue(value);

            while (queue.Count > max)
            {
                queue.Dequeue();
            }
        }
    }

    public sealed class KickBeatDetectorSettings
    {

        /// <summary>
        /// Controls transient responsiveness.
        /// Lower = sharper/more reactive
        /// Higher = smoother/more stable
        /// </summary>
        public float FluxSmoothing { get; set; } = 0.35f;

        /// <summary>
        /// Controls bass body smoothing.
        /// </summary>
        public float EnergySmoothing { get; set; } = 0.25f;

        //--------------------------------------------------
        // Adaptive thresholding
        //--------------------------------------------------

        /// <summary>
        /// Required transient strength above average.
        /// Higher = cleaner but less sensitive.
        /// </summary>
        public float FluxThresholdMultiplier { get; set; } = 1.7f;

        /// <summary>
        /// Required kick body strength above average.
        /// </summary>
        public float EnergyThresholdMultiplier { get; set; } = 1.25f;

        //--------------------------------------------------
        // Transient quality
        //--------------------------------------------------

        /// <summary>
        /// Higher values require sharper attacks.
        /// Lower values react more to sustained bass.
        /// </summary>
        public float SharpnessNormalization { get; set; } = 1.5f;

        //--------------------------------------------------
        // Peak normalization
        //--------------------------------------------------

        /// <summary>
        /// Adaptive peak decay speed.
        /// Lower = faster adaptation.
        /// Higher = more stable scaling.
        /// </summary>
        public float PeakDecay { get; set; } = 0.99f;

        //--------------------------------------------------
        // Envelope smoothing
        //--------------------------------------------------

        /// <summary>
        /// Rise speed.
        /// Higher = snappier attack.
        /// </summary>
        public float AttackSmoothing { get; set; } = 0.65f;

        /// <summary>
        /// Fall speed.
        /// Lower = longer visual decay.
        /// </summary>
        public float ReleaseSmoothing { get; set; } = 0.12f;

        //--------------------------------------------------
        // Adaptive averaging
        //--------------------------------------------------

        /// <summary>
        /// Moving average history size.
        /// Lower = more reactive
        /// Higher = more stable
        /// </summary>
        public int HistorySize { get; set; } = 5;
    }
}
