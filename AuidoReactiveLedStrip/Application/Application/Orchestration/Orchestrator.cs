using Application.Audio.Service;
using Application.Coloring.Remapping.Service;
using Application.Coloring.Service;
using Application.Domain;
using Application.Effect.Service;
using Application.Looper;
using Application.RuntimeSettings;
using Application.Visualization;
using Application.Visualization.Led;
using System.Drawing;

namespace Application.Application.Orchestration
{
    public class Orchestrator : ILooperConsumer, IOrchestrator
    {
        private readonly IAudioService audioService;
        private readonly IEffectService effectService;
        private readonly IColorService colorService;
        private readonly IRemapService remapService;
        private readonly IVisualizer ledVisualizer;
        private readonly ILooper looper;

        private IVisualizer? screenVisualizer;
        private bool isLedVisualizerRunning = false;

        private int invalidFrameSleepTime = 100;
        private bool isRunning = false;
        private Thread? worker = null;
        private Color[]? currentColors = null;

        public Orchestrator(
            IAudioService audioService,
            IEffectService effectService,
            IColorService colorService,
            IRemapService remapService,
            Ws281xLedVisualizer ledVisualizer,
            ILooperFactory looperFactory
        )
        {
            this.audioService = audioService;
            this.effectService = effectService;
            this.colorService = colorService;
            this.remapService = remapService;
            this.ledVisualizer = ledVisualizer;
            this.looper = looperFactory.GetLooper();
            this.looper.SetConsumer(this);
        }

        public void SetCurrentScreen(IVisualizer screenVisualizer)
        {
            this.screenVisualizer = screenVisualizer;
        }

        public void OnSettingsChanged(StaticSettings staticSettings, DynamicSettings dynamicSettings)
        {
            this.invalidFrameSleepTime = staticSettings.InvalidFrameSleepTime;
            this.effectService.SetEffectMode(dynamicSettings.EffectMode);
            this.audioService.SetAudioMode(this.effectService.GetRequiredAudioMode());
            this.colorService.SetColorMode(dynamicSettings.ColorMode);
            this.HandleLedVisualizer(staticSettings);
        }

        private void HandleLedVisualizer(StaticSettings staticSettings)
        {
            if (this.isLedVisualizerRunning == staticSettings.UseLedVisualization)
                return;

            if (staticSettings.UseLedVisualization)
            {
                Console.WriteLine("Starting led visualization");
                this.ledVisualizer.Start();
                this.isLedVisualizerRunning = true;
            }
            else
            {
                Console.WriteLine("Stopping led visualization");
                this.ledVisualizer.Stop();
                this.ledVisualizer.Dispose();
                this.isLedVisualizerRunning = false;
            }
        }

        public void OnTick()
        {
            LedStrip? ledStrip = effectService.GetRenderedLedStrip();

            if (ledStrip != null)
            {
                this.colorService.ColorizeLedStrip(ledStrip);
                this.currentColors = this.remapService.RemapColors(ledStrip);
            }
            else
            {
                this.currentColors = null;
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
            this.ledVisualizer.Stop();
            this.ledVisualizer.Dispose();
            this.audioService.SetAudioMode(AudioServiceMode.None);
            this.isRunning = false;
        }

        private void StartLooper()
        {
            this.looper.StartLooper();
        }

        public void OnBeforeTick()
        {
            if (this.currentColors != null)
            {
                this.screenVisualizer?.UpdateColors(this.currentColors);
                this.ledVisualizer.UpdateColors(this.currentColors);
            }
        }
    }
}
