using Application.Settings;
using Microsoft.Extensions.Options;

namespace Application.Looper
{
    public class LinuxLooper : BaseLooper
    {
        public LinuxLooper(
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

            double elapsed = (sw.Elapsed - frameStart).TotalSeconds;
            double remaining = base.frameTime - elapsed;

            if (remaining > 0)
            {
                Thread.Sleep((int)(remaining * 1000));
            }

            var frameEnd = sw.Elapsed;
            base.PrintFrameTimes(frameStart, processTime, frameEnd);
        }
    }
}
