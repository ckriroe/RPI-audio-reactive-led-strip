using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

namespace Application.Visualization.Screen
{
    public class ScreenVisualizerFactory : IScreenVisualizerFactory
    {
        public IScreenVisualizer Create(int initialWidth, int initialHeight)
        {
            var settings = new NativeWindowSettings()
            {
                ClientSize = new OpenTK.Mathematics.Vector2i(initialWidth, initialHeight),
                Title = "Screen Visualizer",
                API = ContextAPI.OpenGL,
                APIVersion = new Version(3, 3),
                Flags = ContextFlags.Default,
                Profile = ContextProfile.Compatability
            };

            return new ScreenVisualizer(GameWindowSettings.Default, settings);
        }
    }
}
