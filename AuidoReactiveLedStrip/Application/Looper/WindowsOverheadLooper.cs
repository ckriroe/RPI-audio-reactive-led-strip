using Application.Domain;
using Application.Settings;
using Microsoft.Extensions.Options;

namespace Application.Looper
{
    public class WindowsOverheadLooper : BaseLooper
    {
        private const double ThreadSleepAvgDiscrepency = 0.011;

        public WindowsOverheadLooper(
            IOptionsMonitor<StaticSettings> staticSettings,
            IOptionsMonitor<DynamicSettings> dynamicSettings
        ) : base(staticSettings, dynamicSettings)
        { }

        protected override void RunLooperLogic()
        {
            var frameStart = sw.Elapsed;

            base.TryToReloadSettings();
            base.looperConsumer?.OnTick();

            var processEnd = sw.Elapsed;

            double processTime = (processEnd - frameStart).TotalSeconds;

            while (true)
            {
                var elapsed = (sw.Elapsed - frameStart).TotalSeconds;
                double remaining = base.frameTime - elapsed;

                if (remaining <= 0)
                    break;

                if (remaining > ThreadSleepAvgDiscrepency)
                {
                    Thread.Sleep((int)((remaining - ThreadSleepAvgDiscrepency) * 1000));
                }
                else
                {
                    Thread.SpinWait(50);
                }
            }

            var frameEnd = sw.Elapsed;

            base.PrintFrameTimes(frameStart, processTime, frameEnd);
        }
    }
}
