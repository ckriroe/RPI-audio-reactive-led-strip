using Application.LedStripRendering;
using Application.Looper;
using Application.RuntimeSettings;
using Application.Visualization;
using Application.Visualization.Led;
using System.Drawing;

namespace Application.Application.Orchestration
{
    public class Orchestrator : ILooperConsumer, IOrchestrator
    {
        private readonly IVisualizer ledVisualizer;
        private readonly ILedStripRenderer ledStripRenderer;
        private readonly ILooper looper;

        private IVisualizer? screenVisualizer;
        private bool isLedVisualizerRunning = false;

        private int invalidFrameSleepTime = 100;
        private bool isRunning = false;
        private Thread? worker = null;
        private Color[]? currentColors = null;

        public Orchestrator(
            Ws281xLedVisualizer ledVisualizer,
            ILooperFactory looperFactory,
            ILedStripRenderer ledStripRenderer
        )
        {
            this.ledVisualizer = ledVisualizer;
            this.looper = looperFactory.GetLooper();
            this.looper.SetConsumer(this);
            this.ledStripRenderer = ledStripRenderer;
        }

        public void SetCurrentScreen(IVisualizer screenVisualizer)
        {
            this.screenVisualizer = screenVisualizer;
        }

        public void OnSettingsChanged(StaticSettings staticSettings, DynamicPresetSettings dynamicSettings)
        {
            this.invalidFrameSleepTime = staticSettings.InvalidFrameSleepTime;
            this.ledStripRenderer.ApplySettings(staticSettings, dynamicSettings.Presets.First().EffectSettings);
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
            this.currentColors = this.ledStripRenderer.RenderLedStrip();
            if (currentColors == null)
                Thread.Sleep(this.invalidFrameSleepTime);
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
            this.ledStripRenderer.Dispose();
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
