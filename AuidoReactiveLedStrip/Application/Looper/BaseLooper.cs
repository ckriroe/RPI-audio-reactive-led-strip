using Application.RuntimeSettings;
using Microsoft.Extensions.Options;
using System.Diagnostics;

namespace Application.Looper
{
    public abstract class BaseLooper : ILooper
    {
        protected readonly IOptionsMonitor<StaticSettings> staticSettings;
        protected readonly IOptionsMonitor<DynamicPresetSettings> dynamicSettings;
        private readonly object lck = new object();

        private volatile bool isLooperRunning = false;
        private int fps = 10;
        private bool printFrameTimes = false;

        private IDisposable? staticSettingsSubscription;
        private IDisposable? dynamicSettingsSubscription;
        private StaticSettings? currentStaticSettings;
        private DynamicPresetSettings? currentDynamicSettings;
        private bool shouldReloadSettings = false;

        protected double frameTime = 100.0;
        protected Stopwatch sw = Stopwatch.StartNew();
        protected ILooperConsumer? looperConsumer;

        public BaseLooper(IOptionsMonitor<StaticSettings> staticSettings, IOptionsMonitor<DynamicPresetSettings> dynamicSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicSettings;
        }

        public void SetConsumer(ILooperConsumer looperConsumer)
        {
            this.looperConsumer = looperConsumer;
        }

        public void StartLooper()
        {
            if (this.isLooperRunning)
                return;

            this.isLooperRunning = true;
            this.staticSettingsSubscription?.Dispose();
            this.staticSettingsSubscription = this.staticSettings.OnChange((staticSettings) =>
            {
                lock (this.lck)
                {
                    this.currentStaticSettings = staticSettings;
                    this.shouldReloadSettings = true;
                }
            });

            this.dynamicSettingsSubscription?.Dispose();
            this.dynamicSettingsSubscription = this.dynamicSettings.OnChange((dynamicSettings) =>
            {
                lock (this.lck)
                {
                    this.currentDynamicSettings = dynamicSettings;
                    this.shouldReloadSettings = true;
                }
            });

            this.currentStaticSettings = this.staticSettings.CurrentValue;
            this.currentDynamicSettings = this.dynamicSettings.CurrentValue;
            this.shouldReloadSettings = true;

            this.sw.Restart();
            while (this.isLooperRunning)
            {
                RunLooperLogic();
            }
        }

        public void StopLooper()
        {
            this.staticSettingsSubscription?.Dispose();
            this.dynamicSettingsSubscription?.Dispose();
            this.isLooperRunning = false;
        }

        protected void TryToReloadSettings()
        {
            if (!this.shouldReloadSettings)
                return;

            lock (this.lck)
            {
                if (!this.shouldReloadSettings)
                    return;

                DynamicPresetSettings? dynamicSettings = this.currentDynamicSettings;
                StaticSettings? staticSettings = this.currentStaticSettings;
                if (dynamicSettings == null || staticSettings == null)
                    return;

                this.printFrameTimes = staticSettings.PrintFrameTimes;
                this.fps = staticSettings.Fps;
                this.frameTime = 1.0 / fps;

                this.looperConsumer?.OnSettingsChanged(staticSettings, dynamicSettings);
                this.shouldReloadSettings = false;
            }
        }

        protected void PrintFrameTimes(TimeSpan frameStart, double processTime, TimeSpan frameEnd)
        {
            if (!this.printFrameTimes)
                return;

            double totalFrameTime = (frameEnd - frameStart).TotalSeconds;
            double theoreticalFps = processTime > 0
                ? 1.0 / processTime
                : double.PositiveInfinity;

            Console.WriteLine(
                $"Frame: {totalFrameTime * 1000:F2} ms | " +
                $"Process: {processTime * 1000:F2} ms | " +
                $"Theo FPS: {theoreticalFps:F1}"
            );
        }

        protected abstract void RunLooperLogic();
    }
}
