using Application.Audio.Service;
using Application.Coloring.Service;
using Application.Domain;
using Application.Effect.Service;
using Application.Looper;
using Application.Settings;
using Application.Visualization.Screen;

namespace Application.Application.Orchestration
{
    public class Orchestrator : ILooperConsumer, IOrchestrator
    {
        private readonly IAudioService audioService;
        private readonly IEffectService effectService;
        private readonly IColorService colorService;
        private readonly ILooper looper;

        private IScreenVisualizer? screenVisualizer;
        private int invalidFrameSleepTime = 100;
        private bool isRunning = false;
        private Thread? worker = null;

        public Orchestrator(
            IAudioService audioService,
            IEffectService effectService,
            IColorService colorService,
            ILooper looper
        )
        {
            this.audioService = audioService;
            this.effectService = effectService;
            this.colorService = colorService;
            this.looper = looper;
            this.looper.SetConsumer(this);
        }

        public void SetCurrentScreen(IScreenVisualizer screenVisualizer)
        {
            this.screenVisualizer = screenVisualizer;
        }

        public void OnSettingsChanged(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            this.invalidFrameSleepTime = staticSettings.InvalidFrameSleepTime;
            this.effectService.SetEffectMode(dynamicSettings.EffectMode);
            this.audioService.SetAudioMode(this.effectService.GetRequiredAudioMode());
            this.colorService.SetColorMode(dynamicSettings.ColorMode);
        }

        public void OnTick()
        {
            LedStrip? ledStrip = effectService.GetRenderedLedStrip();

            if (ledStrip != null)
            {
                this.colorService.ColorizeLedStrip(ledStrip);
                this.screenVisualizer?.UpdateColors(ledStrip.LedPixels.Select(lp => lp.Color).ToArray());
                // TODO color actual leds
            }
            else
            {
                Thread.Sleep(this.invalidFrameSleepTime);
            }
        }

        public void Start()
        {
            if (this.isRunning)
                return;

            this.worker = new Thread(StartLooper);
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
