using Application.Audio.Receiver;
using Application.Audio.ValueProvider;
using Application.Audio.FftTransformer;
using Application.Domain;
using Application.Settings;
using Microsoft.Extensions.Options;

namespace Application.Audio.Service
{
    public class AudioService : IAudioService
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettingsMonitor;
        private readonly IOptionsMonitor<DynamicSettings> dynamicSettingsMonitor;
        private readonly IAudioReceiver audioReceiver;
        private readonly IAudioFftTransformer audioFftTransformer;
        private readonly AudioFftDataProvider audioFftDataProvider;
        private readonly SimpleAudioValueProvider simpleAudioValueProvider;
        private readonly MovingMaxAudioValueProvider movingMaxAudioValueProvider;

        private AudioServiceMode currentAudioMode = AudioServiceMode.None;
        private IAudioDataProvider? currentDataProvider = null;
        private bool isRunning = false;

        public AudioService
        (
            IOptionsMonitor<StaticSettings> staticSettingsMonitor,
            IOptionsMonitor<DynamicSettings> dynamicSettingsMonitor,
            IAudioReceiver audioReceiver,
            IAudioFftTransformer audioFftTransformer,
            AudioFftDataProvider audioFftDataProvider,
            SimpleAudioValueProvider simpleAudioValueProvider,
            MovingMaxAudioValueProvider movingMaxAudioValueProvider
        )
        {
            this.staticSettingsMonitor = staticSettingsMonitor;
            this.dynamicSettingsMonitor = dynamicSettingsMonitor;
            this.audioReceiver = audioReceiver;
            this.audioFftTransformer = audioFftTransformer;
            this.audioFftDataProvider = audioFftDataProvider;
            this.simpleAudioValueProvider = simpleAudioValueProvider;
            this.movingMaxAudioValueProvider = movingMaxAudioValueProvider;

            this.staticSettingsMonitor.OnChange(_ => this.ApplySettings());
            this.dynamicSettingsMonitor.OnChange(_ => this.ApplySettings());

            this.ApplySettings();
        }

        public void SetAudioMode(AudioServiceMode audioMode)
        {
            if (this.currentAudioMode == audioMode)
                return;

            this.currentAudioMode = audioMode;
            if (audioMode == AudioServiceMode.None)
            {
                this.StopAudioProcessing();
                this.currentDataProvider = null;
            } else
            {
                if (audioMode == AudioServiceMode.Fft)
                    this.currentDataProvider = this.audioFftDataProvider;
                else if (audioMode == AudioServiceMode.Simple)
                    this.currentDataProvider = this.simpleAudioValueProvider;
                else if (audioMode == AudioServiceMode.MovingMax)
                    this.currentDataProvider = this.movingMaxAudioValueProvider;

                if (this.currentDataProvider != null)
                    this.StartAudioProcessing();
            }
        }

        public float[]? GetCurrentFftData()
        {
            return this.currentDataProvider?.GetCurrentFftData();
        }

        public float? GetCurrentAudioValue()
        {
            if (this.currentDataProvider is BaseAudioValueProvider bavp)
                return bavp.GetAudioValue();

            return null;
        }

        private void StartAudioProcessing()
        {
            if (this.isRunning)
                return;

            try
            {
                this.audioReceiver.StartAudioStream(data =>
                {
                    float[]? fftData = this.audioFftTransformer.ProcessAudioSamples(data);
                    if (fftData != null)
                        this.currentDataProvider?.SetNewFftData(fftData);
                });

                this.isRunning = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to start audio processing: " + ex);
            }
        }

        private void StopAudioProcessing()
        {
            if (!this.isRunning)
                return;

            try
            {
                this.audioReceiver.StopAudioStream();
                this.isRunning = false;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to stop audio processing: " + ex);
            }
        }

        private void ApplySettings()
        {
            var staticSettings = this.staticSettingsMonitor.CurrentValue;
            this.audioReceiver.Initialize
            (
                staticSettings.AudioDeviceId,
                staticSettings.Channels,
                staticSettings.FftSize,
                staticSettings.SampleRate,
                () =>
                {
                    this.audioFftTransformer.Initialize(staticSettings.Channels);
                }
            );

            var dynamicSettings = this.dynamicSettingsMonitor.CurrentValue;
            this.audioFftDataProvider.Initialize(new BaseAudioDataProviderSettings()
            {
                FftSize = staticSettings.FftSize,
                SampleRate = staticSettings.SampleRate,
                MinFrequency = dynamicSettings.MinFreq,
                MaxFrequency = dynamicSettings.MaxFreq
            });

            this.simpleAudioValueProvider.Initialize(new BaseAudioValueProviderSettings()
            {
                FftSize = staticSettings.FftSize,
                SampleRate = staticSettings.SampleRate,
                MinFrequency = dynamicSettings.MinFreq,
                MaxFrequency = dynamicSettings.MaxFreq,
                MinFrequencyAmplitude = dynamicSettings.MinFreqAmplitude,
                MaxFrequencyAmplitude = dynamicSettings.MaxFreqAmplitude
            });

            this.movingMaxAudioValueProvider.Initialize(new MovingMaxAudioValueProviderSettings()
            {
                FftSize = staticSettings.FftSize,
                SampleRate = staticSettings.SampleRate,
                MinFrequency = dynamicSettings.MinFreq,
                MaxFrequency = dynamicSettings.MaxFreq,
                MinFrequencyAmplitude = dynamicSettings.MinFreqAmplitude,
                MaxFrequencyAmplitude = dynamicSettings.MaxFreqAmplitude,
                LastExtraOrdanarySampleBufferSize = dynamicSettings.MeanValueBufferSize,
                BelowMinFreqAmplitudeFunctionFactor = staticSettings.BelowMinFreqAmplitudeFunctionFactor,
                MaxFreqAmplitudeIncreaseRatio = staticSettings.MaxFreqAmplitudeIncreaseRatio,
                MaxFreqAmplitudeDecreaseRatio = staticSettings.MaxFreqAmplitudeDecreaseRatio,
                MaxFreqAmplitudeTTL = staticSettings.MaxFreqAmplitudeTTL,
                MaxFreqAmplitudeProlongerThreshholdPercent = staticSettings.MaxFreqAmplitudeProlongerThreshholdPercent,
                MaxFreqAmplitudeDecayRate = staticSettings.MaxFreqAmplitudeDecayRate,
                PercentDiffFromMaxToBeExtraOrdanary = dynamicSettings.MeanValueThreshold
            });
        }
    }
}
