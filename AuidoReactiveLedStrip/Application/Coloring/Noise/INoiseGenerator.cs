using Application.Settings;

namespace Application.Coloring.Noise
{
    public interface INoiseGenerator
    {
        float GetSmoothNoise(int index, int length, DynamicSettings dynamicSettings);
    }
}
