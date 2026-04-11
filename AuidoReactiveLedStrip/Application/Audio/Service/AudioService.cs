using Application.Audio.Receiver;
using Application.Audio.ValueProvider;
using Application.Audio.FftTransformer;
using Application.Domain;
using Application.RuntimeSettings;
using Microsoft.Extensions.Options;

namespace Application.Audio.Service
{
    public class AudioService : IAudioService
    {
        private readonly Guid audioServiceIdentifier = Guid.NewGuid();
        private readonly IAudioReceiver audioReceiver;
        private readonly IAudioFftTransformer audioFftTransformer;
        private readonly AudioFftDataProvider audioFftDataProvider;
        private readonly SimpleAudioValueProvider simpleAudioValueProvider;
        private readonly MovingMaxAudioValueProvider movingMaxAudioValueProvider;

        private AudioServiceMode currentAudioMode = AudioServiceMode.None;
        private IAudioDataProvider? currentDataProvider = null;
        private bool isRunning = false;
        private int? currentBufferSize = null;

        public AudioService
        (
            IAudioReceiver audioReceiver,
            IAudioFftTransformer audioFftTransformer,
            AudioFftDataProvider audioFftDataProvider,
            SimpleAudioValueProvider simpleAudioValueProvider,
            MovingMaxAudioValueProvider movingMaxAudioValueProvider
        )
        {
            this.audioReceiver = audioReceiver;
            this.audioFftTransformer = audioFftTransformer;
            this.audioFftDataProvider = audioFftDataProvider;
            this.simpleAudioValueProvider = simpleAudioValueProvider;
            this.movingMaxAudioValueProvider = movingMaxAudioValueProvider;
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

        public void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings)
        {
            this.audioReceiver.ApplyStaticSettings(staticSettings);
            if (this.currentBufferSize != dynamicSettings.FftSize)
            {
                this.currentBufferSize = dynamicSettings.FftSize;
                this.TryEnableAudioProcessing();
            }

            this.audioFftDataProvider.ApplySettings(staticSettings, dynamicSettings);
            this.simpleAudioValueProvider.ApplySettings(staticSettings, dynamicSettings);
            this.movingMaxAudioValueProvider.ApplySettings(staticSettings, dynamicSettings);
        }

        public void SetAudioMode(AudioServiceMode audioMode)
        {
            if (this.currentAudioMode == audioMode)
                return;

            this.currentDataProvider?.SetActive(false);
            this.currentAudioMode = audioMode;
            if (audioMode == AudioServiceMode.None)
            {
                this.StopAudioProcessing();
                this.currentDataProvider = null;
            } 
            else
            {
                Console.WriteLine("Set audio mode: " + audioMode.ToString());
                if (audioMode == AudioServiceMode.Fft)
                    this.currentDataProvider = this.audioFftDataProvider;
                else if (audioMode == AudioServiceMode.Simple)
                    this.currentDataProvider = this.simpleAudioValueProvider;
                else if (audioMode == AudioServiceMode.MovingMax)
                    this.currentDataProvider = this.movingMaxAudioValueProvider;
                
                this.TryEnableAudioProcessing();
            }
        }

        private void TryEnableAudioProcessing()
        {
            if (this.currentDataProvider != null)
            {
                this.currentDataProvider.SetActive(true);
                this.StartAudioProcessing();
            }
        }

        private void StartAudioProcessing()
        {
            if (this.currentBufferSize == null)
                return;

            try
            {
                this.audioReceiver.RegisterAudioConsumer(this.audioServiceIdentifier, this.currentBufferSize.Value, (data, channels) =>
                {
                    float[] fftData = this.audioFftTransformer.ProcessAudioSamples(data, channels);
                    this.currentDataProvider?.SetNewFftData(fftData);
                });

                this.isRunning = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to start audio processing: " + ex);
                this.isRunning = false;
            }
        }

        private void StopAudioProcessing()
        {
            if (!this.isRunning)
                return;

            try
            {
                this.audioReceiver.UnregisterAudioConsumer(this.audioServiceIdentifier);
                this.isRunning = false;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to stop audio processing: " + ex);
            }
        }
    }
}
