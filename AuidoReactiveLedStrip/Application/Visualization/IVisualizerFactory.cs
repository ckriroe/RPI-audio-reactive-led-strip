namespace Application.Visualization
{
    public interface IVisualizerFactory
    {
        IVisualizer Create(int initialWidth, int initialHeight);
    }
}