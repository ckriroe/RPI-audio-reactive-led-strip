using Application.RuntimeSettings;
using Microsoft.Extensions.Options;

namespace Application.Looper
{
    public class StaticLooperFactory : ILooperFactory
    {
        private readonly ILooper looper;

        public StaticLooperFactory(IOptionsMonitor<StaticSettings> staticSettings, IOptionsMonitor<DynamicPresetSettings> dynamicSettings)
        {
            if (staticSettings.CurrentValue.AccurateSleeping)
            {
                this.looper = new AccurateOverheadLooper(staticSettings, dynamicSettings);
            }
            else
            {
                this.looper = new SleepLooper(staticSettings, dynamicSettings);
            }
        }

        public ILooper GetLooper()
        {
            return this.looper;
        }
    }
}
