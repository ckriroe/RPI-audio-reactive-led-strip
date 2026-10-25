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
                Title = GetWindowTitle(),
                API = ContextAPI.OpenGL,
                APIVersion = new Version(3, 0),
                Flags = ContextFlags.Default,
                Profile = ContextProfile.Any
            };

            return new OpenTkScreenVisualizer(GameWindowSettings.Default, settings, this.staticSettings);
        }

        private static string GetWindowTitle()
        {
            const string baseTitle = "Screen Visualizer";
            try
            {
                string path = Path.Combine(AppContext.BaseDirectory, "version.txt");
                if (!File.Exists(path))
                {
                    path = "version.txt";
                }

                if (File.Exists(path))
                {
                    string version = File.ReadAllText(path).Trim();
                    if (!string.IsNullOrWhiteSpace(version))
                    {
                        return $"{baseTitle} {version}";
                    }
                }
            }
            catch
            {
                // Optional file; fallback to baseTitle
            }

            return baseTitle;
        }
    }
}
