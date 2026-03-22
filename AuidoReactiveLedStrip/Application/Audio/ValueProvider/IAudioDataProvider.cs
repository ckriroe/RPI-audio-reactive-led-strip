using Application.Settings;

namespace Application.Audio.ValueProvider
{
    public interface IAudioDataProvider
    {
        void Initialize(StaticSettings staticSettings, DynamicSettings dynamicSettings);

        void SetNewFftData(float[] fftData);

        float[]? GetCurrentFftData();
    }
}
