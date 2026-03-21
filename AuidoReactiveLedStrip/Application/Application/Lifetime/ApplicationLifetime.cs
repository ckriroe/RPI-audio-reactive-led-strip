using Application.Application.Service;

namespace Application.Application.Lifetime
{
    public class ApplicationLifetime : IApplicationLifetime
    {
        private Thread? lifetimeThread = null;
        private IApplicationService? currentApplication;

        public void StartLifetime(IApplicationService application)
        {
            if (this.lifetimeThread != null)
                return;

            this.currentApplication = application;
            this.lifetimeThread = new Thread(() =>
            {
                Console.WriteLine("Press any key to close this application...");
                Console.ReadKey();
                Console.WriteLine("\nShutting down...");
                this.currentApplication?.StopApplication();
            });

            this.lifetimeThread.Start();
        }

        public void StopLifetime()
        {
            this.lifetimeThread?.Join();
            this.lifetimeThread = null;
        }
    }
}
