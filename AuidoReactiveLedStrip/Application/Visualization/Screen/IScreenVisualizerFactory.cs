namespace Application.Visualization.Screen
{
    public interface IScreenVisualizerFactory
    {
        IScreenVisualizer Create(int initialWidth, int initialHeight);
    }
}