using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

namespace Application.Visualization.Screen
{
    public class OpenTkScreenVisualizerFactory : IVisualizerFactory
    {
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

            return new OpenTkScreenVisualizer(GameWindowSettings.Default, settings);
        }
    }
}
