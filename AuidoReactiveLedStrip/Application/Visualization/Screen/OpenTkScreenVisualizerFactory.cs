using Application.RuntimeSettings;
using Microsoft.Extensions.Options;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

namespace Application.Visualization.Screen
{
    public class OpenTkScreenVisualizerFactory : IVisualizerFactory
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;

        public OpenTkScreenVisualizerFactory(IOptionsMonitor<StaticSettings> staticSettings)
        {
            this.staticSettings = staticSettings;
        }

        public IVisualizer Create(int initialWidth, int initialHeight)
        {
            var settings = new NativeWindowSettings()
            {
                ClientSize = new OpenTK.Mathematics.Vector2i(initialWidth, initialHeight),
                Title = "Screen Visualizer",
                API = ContextAPI.OpenGL,
                APIVersion = new Version(3, 0),
                Flags = ContextFlags.Default,
                Profile = ContextProfile.Any
            };

            return new OpenTkScreenVisualizer(GameWindowSettings.Default, settings, this.staticSettings);
        }
    }
}
