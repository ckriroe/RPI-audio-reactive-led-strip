using Application.Application.Service;

namespace Application.Application.Lifetime
{
    public interface IApplicationLifetime
    {
        void StartLifetime(IApplicationService application);
        void StopLifetime();
    }
}