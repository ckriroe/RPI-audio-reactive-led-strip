using Application.RuntimeSettings;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Looper
{
    public abstract class BaseLooper : ILooper
    {
        protected readonly IOptionsMonitor<StaticSettings> staticSettings;
        protected readonly IOptionsMonitor<DynamicSettings> dynamicSettings;

        private volatile bool isLooperRunning = false;
        private bool isInitialFrame = true;
        private int fps = 10;
        private int reloadSettingsAfterMs = 100;
        private bool printFrameTimes = false;

        protected double frameTime = 100.0;
        protected Stopwatch sw = Stopwatch.StartNew();
        protected ILooperConsumer? looperConsumer;
        protected TimeSpan? lastSettingsReload = null;

        public BaseLooper(IOptionsMonitor<StaticSettings> staticSettings, IOptionsMonitor<DynamicSettings> dynamicSettings)
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
            this.sw.Restart();
            while (this.isLooperRunning)
            {
                RunLooperLogic();
            }
        }

        public void StopLooper()
        {
            this.isLooperRunning = false;
        }

        protected void TryToReloadSettings()
        {
            if ((this.lastSettingsReload != null && (sw.Elapsed - this.lastSettingsReload.Value).TotalMilliseconds > this.reloadSettingsAfterMs) || this.isInitialFrame)
            {
                DynamicSettings dynamicSettings = this.dynamicSettings.CurrentValue;
                StaticSettings staticSettings = this.staticSettings.CurrentValue;

                this.printFrameTimes = staticSettings.PrintFrameTimes;
                this.reloadSettingsAfterMs = staticSettings.ReloadSettingsAfterMs;
                this.fps = dynamicSettings.Fps;
                this.frameTime = 1.0 / fps;

                this.looperConsumer?.OnSettingsChanged(staticSettings, dynamicSettings);
                isInitialFrame = false;
                this.lastSettingsReload = sw.Elapsed;
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
