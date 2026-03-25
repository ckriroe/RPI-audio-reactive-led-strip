using Application.Application.Service;
using Application.RuntimeSettings;
using Microsoft.Extensions.Options;

namespace Application.Application.Lifetime
{
    public class ApplicationLifetime : IApplicationLifetime
    {
        private readonly bool runIndefinitely;

        private Thread? lifetimeThread = null;
        private IApplicationService? currentApplication;

        public ApplicationLifetime(IOptions<StaticSettings> options)
        {
            this.runIndefinitely = options.Value.RunIndefinitely;
        }

        public void StartLifetime(IApplicationService application)
        {
            if (this.lifetimeThread != null)
                return;

            this.currentApplication = application;
            this.lifetimeThread = new Thread(() =>
            {
                if (this.runIndefinitely)
                {
                    while (true)
                    {
                        Thread.Sleep(10000);
                    }
                }
                else
                {
                    Console.WriteLine("Press any key to close this application...");
                    Console.ReadKey();
                    Console.WriteLine("\nShutting down...");
                }

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
