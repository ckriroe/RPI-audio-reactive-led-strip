using Application.Settings;
using Application.Visualization.Screen;

namespace Application.Application.Orchestration
{
    public interface IOrchestrator
    {
        void OnSettingsChanged(StaticSettings staticSettings, DynamicSettings dynamicSettings);
        void OnTick();
        void SetCurrentScreen(IScreenVisualizer screen);
        void Start();
        void Stop();
    }
}