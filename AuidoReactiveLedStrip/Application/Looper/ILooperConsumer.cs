using Application.RuntimeSettings;

namespace Application.Looper
{
    public interface ILooperConsumer
    {
        public void OnSettingsChanged(StaticSettings staticSettings, DynamicPresetSettings dynamicSettings);
        public void OnTick();
        public void OnBeforeTick();
        public void OnMetricsUpdated(LooperPerformanceMetrics metrics);
    }
}
