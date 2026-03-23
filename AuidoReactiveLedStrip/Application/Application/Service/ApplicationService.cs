using Application.Application.Orchestration;
using Application.Settings;
using Application.Visualization.Screen;
using Microsoft.Extensions.Options;
using System.Diagnostics;

namespace Application.Application.Service
{
    public class ApplicationService : IApplicationService
    {

        private readonly IScreenVisualizerFactory screenVisualizerFactory;
        private readonly IOptionsMonitor<StaticSettings> staticSettingsMonitor;
        private readonly IOrchestrator orchestrator;
        private readonly Lifetime.IApplicationLifetime applicationLifetime;
        private readonly IDisposable? settingsSubscription;

        private volatile bool isRunnging = false;
        private int prevGuiWidth = 0;
        private int prevGuiHeight = 0;
        private IScreenVisualizer? currentScreenVisualizer = null;


        public ApplicationService(
            IScreenVisualizerFactory screenVisualizerFactory,
            IOptionsMonitor<StaticSettings> staticSettingsMonitor,
            IOrchestrator orchestrator,
            Lifetime.IApplicationLifetime applicationLifetime
        )
        {
            this.screenVisualizerFactory = screenVisualizerFactory;
            this.staticSettingsMonitor = staticSettingsMonitor;
            this.orchestrator = orchestrator;
            this.applicationLifetime = applicationLifetime;
            this.settingsSubscription = this.staticSettingsMonitor.OnChange(this.OnStaticSettingsChanged);
        }

        public void StartApplication()
        {
            if (this.isRunnging)
                return;

            this.isRunnging = true;
            this.orchestrator.Start();
            this.applicationLifetime.StartLifetime(this);
            while (this.isRunnging)
            {
                StaticSettings staticSettings = this.staticSettingsMonitor.CurrentValue;
                if (staticSettings.UseGuiVisualization)
                {
                    this.prevGuiWidth = staticSettings.GuiWidth;
                    this.prevGuiHeight = staticSettings.GuiHeight;
                    this.currentScreenVisualizer = this.screenVisualizerFactory.Create(staticSettings.GuiWidth, staticSettings.GuiHeight);
                    this.orchestrator.SetCurrentScreen(this.currentScreenVisualizer);
                    Console.WriteLine($"Displaying visualization (Width: {staticSettings.GuiWidth}px, Height: {staticSettings.GuiHeight}px)");
                    this.currentScreenVisualizer.Run();
                    Console.WriteLine("Closed visualization");
                    this.currentScreenVisualizer.Dispose();
                    this.currentScreenVisualizer = null;
                    this.ForceCloseWindow(staticSettings);
                }
                else
                {
                    Thread.Sleep(staticSettings.MainThreadSettingsCheckIntervalMs);
                }
            }

            this.applicationLifetime.StopLifetime();
            this.orchestrator.Stop();
        }

        public void OnStaticSettingsChanged(StaticSettings staticSettings)
        {
            if (this.currentScreenVisualizer != null && (
                !staticSettings.UseGuiVisualization ||
                staticSettings.GuiWidth != this.prevGuiWidth ||
                staticSettings.GuiHeight != this.prevGuiHeight
            ) && this.isRunnging)
            {
                // Window gets closed until its configured again or gets reopened with new size
                this.currentScreenVisualizer?.Close();
            }
        }

        public void StopApplication()
        {
            this.settingsSubscription?.Dispose();
            this.isRunnging = false;
            this.currentScreenVisualizer?.Close();
        }

        private void ForceCloseWindow(StaticSettings staticSettings)
        {
            if (OperatingSystem.IsLinux())
            {
                // Open a pre-clsoed instance of the visualizer to force close the existing window, necessary due to Wayland not handling window close requests correctly
                IScreenVisualizer deadVisualizer = this.screenVisualizerFactory.Create(staticSettings.GuiWidth, staticSettings.GuiHeight);
                deadVisualizer.Close();
                deadVisualizer.Run();
                deadVisualizer.Dispose();
            }
        }
    }
}
