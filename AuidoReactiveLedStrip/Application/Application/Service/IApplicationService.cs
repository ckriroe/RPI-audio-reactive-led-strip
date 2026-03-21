using Application.Settings;

namespace Application.Application.Service
{
    public interface IApplicationService
    {
        void OnStaticSettingsChanged(StaticSettings staticSettings);
        void StartApplication();
        void StopApplication();
    }
}