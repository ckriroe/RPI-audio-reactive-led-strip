using System.Drawing;

namespace Application.Visualization.Screen
{
    public interface IScreenVisualizer : IDisposable
    {
        void Run();
        void Close();
        void UpdateColors(Color[] colors);
    }
}