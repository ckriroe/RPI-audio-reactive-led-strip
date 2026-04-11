using Application.RuntimeSettings;
using Application.Visualization;

namespace Application.Application.Orchestration
{
    public interface IOrchestrator
    {
        void OnTick();
        void SetCurrentScreen(IVisualizer screen);
        void Start();
        void Stop();
    }
}