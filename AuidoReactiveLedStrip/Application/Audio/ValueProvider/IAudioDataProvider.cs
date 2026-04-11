using Application.RuntimeSettings;

namespace Application.Audio.ValueProvider
{
    public interface IAudioDataProvider
    {
        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);

        void SetNewFftData(float[] fftData);

        float[]? GetCurrentFftData();

        void SetActive(bool isActive);
    }
}
