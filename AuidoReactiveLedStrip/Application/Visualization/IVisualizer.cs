using System.Drawing;

namespace Application.Visualization
{
    public interface IVisualizer : IDisposable
    {
        void Start();
        void Stop();
        void UpdateColors(Color[] colors);
    }
}