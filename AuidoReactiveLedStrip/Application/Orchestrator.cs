using Application.Audio.AudioReceiver;
using Application.Audio.AudioValueProvider;
using Application.Audio.FftTransformer;
using Application.Settings;
using Microsoft.Extensions.Options;
using System.Diagnostics;

namespace Application
{
    public class Orchestrator
    {
        private readonly IOptionsMonitor<AudioSettings> audioOptionsMonitor;
        private readonly IAudioReceiver audioReceiver;
        private readonly IAudioFftTransformer audioFftTransformer;
        private readonly IAudioDataProvider audioValueProvider;
        private bool isStarted = false;

        public Orchestrator(
            IOptionsMonitor<AudioSettings> audioOptionsMonitor,
            IAudioReceiver audioReceiver,
            IAudioFftTransformer audioFftTransformer
        ) {
            this.audioOptionsMonitor = audioOptionsMonitor;
            this.audioOptionsMonitor.OnChange(this.SetAudioSettingsChanged);
            this.audioReceiver = audioReceiver;
            this.audioFftTransformer = audioFftTransformer;
            this.SetAudioSettingsChanged(audioOptionsMonitor.CurrentValue);

            this.audioValueProvider = new MovingMaxAudioValueProvider();
            this.audioValueProvider.Initialize(new MovingMaxAudioValueProviderSettings()
            {
                FftSize = audioOptionsMonitor.CurrentValue.FftSize,
                LastExtraOrdanarySampleBufferSize = 30,
                MinFrequency = 0,
                MaxFrequency = 180,
                MinFrequencyAmplitude = 10,
                MaxFrequencyAmplitude = 99999,
                BelowMinFreqAmplitudeFunctionFactor = -0.03f,
                MaxFreqAmplitudeIncreaseRatio = 3,
                MaxFreqAmplitudeDecreaseRatio = 5,
                MaxFreqAmplitudeTTL = 2000,
                MaxFreqAmplitudeProlongerThreshholdPercent = 0.03f,
                MaxFreqAmplitudeDecayRate = 0.003f,
                PercentDiffFromMaxToBeExtraOrdanary = 0.45f,
                SampleRate = audioOptionsMonitor.CurrentValue.SampleRate,
            });
        }

        public void Start()
        {
            if (this.isStarted)
                this.Stop();

            this.StartAudio();
            this.isStarted = true;
        }

        public void Stop()
        {
            if (!this.isStarted)
                return;

            this.StopAudio();
            this.isStarted = false;
        }

        public void SetAudioSettingsChanged(AudioSettings changedAudioSettings)
        {
            this.audioReceiver.Initialize
            (
                changedAudioSettings.AudioDeviceId,
                changedAudioSettings.Channels,
                changedAudioSettings.FftSize,
                changedAudioSettings.SampleRate,
                () =>
                {
                    this.audioFftTransformer.Initialize(changedAudioSettings.Channels);
                }
            );
        }

        private void StartAudio()
        {
            this.audioReceiver.StartAudioStream(data =>
            {
                float[]? fftData = this.audioFftTransformer.ProcessAudioSamples(data);
                if (fftData != null)
                   this.audioValueProvider.SetNewFftData(fftData);
            });
        }

        private void StopAudio()
        {
            this.audioReceiver.StopAudioStream();
        }
    }
}
