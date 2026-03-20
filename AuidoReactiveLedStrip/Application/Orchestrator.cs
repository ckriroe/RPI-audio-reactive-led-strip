using Application.Audio.AudioService;
using Application.Domain;
using Application.Effect.Service;
using Application.Looper;
using Application.Settings;
using Microsoft.Extensions.Options;
using System.Diagnostics;

namespace Application
{
    public class Orchestrator : ILooperConsumer
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;
        private readonly IOptionsMonitor<DynamicSettings> dynamicSettings;
        private readonly IAudioService audioService;
        private readonly IEffectService effectService;
        private readonly ILooper looper;

        private int invalidFrameSleepTime = 100;
        private bool isRunning = false;        
        private Thread? worker = null;

        public Orchestrator(
            IOptionsMonitor<StaticSettings> staticSettings,
            IOptionsMonitor<DynamicSettings> dynamicSettings,
            IAudioService audioService,
            IEffectService effectService,
            ILooper looper
        ) {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;
            this.audioService = audioService;
            this.effectService = effectService;
            this.looper = looper;
            this.looper.SetConsumer(this);
        }

        public void OnSettingsChanged(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            invalidFrameSleepTime = staticSettings.InvalidFrameSleepTime;
            this.effectService.SetEffectMode(dynamicSettings.EffectMode);
            this.audioService.SetAudioMode(this.effectService.GetRequiredAudioMode());
        }

        public void OnTick()
        {
            LedStrip? ledStrip = this.effectService.GetRenderedLedStrip();

            if (ledStrip != null)
            {
                // Do something
            } else
            {
                Thread.Sleep(invalidFrameSleepTime);
            }
        }

        public void Start()
        {
            if (this.isRunning)
                return;

            this.worker = new Thread(this.StartLooper);
            this.worker.Start();

            this.isRunning = true;
        }

        public void Stop()
        {
            if (!this.isRunning)
                return;

            this.looper.StopLooper();
            this.worker?.Join();
            this.audioService.SetAudioMode(AudioServiceMode.None);
            this.isRunning = false;
        }

        private void StartLooper()
        {
            this.looper.StartLooper();
        }
    }
}
