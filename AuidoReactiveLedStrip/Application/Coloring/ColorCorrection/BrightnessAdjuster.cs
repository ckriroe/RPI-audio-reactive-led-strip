using Application.RuntimeSettings;

namespace Application.Coloring.ColorCorrection
{
    public class BrightnessAdjuster : IColorCorrector
    {
        public (float R, float G, float B) ColorCorrect((float R, float G, float B) colors, DynamicSettings dynamicSettings)
        {
            return (
                R: colors.R * dynamicSettings.Brightness,
                G: colors.G * dynamicSettings.Brightness,
                B: colors.B * dynamicSettings.Brightness
            );
        }
    }
}
