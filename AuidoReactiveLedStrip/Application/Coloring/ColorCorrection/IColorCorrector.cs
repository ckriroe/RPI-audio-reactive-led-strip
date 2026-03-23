using Application.Settings;

namespace Application.Coloring.ColorCorrection
{
    public interface IColorCorrector
    {
        (float R, float G, float B) ColorCorrect((float R, float G, float B) colors, DynamicSettings dynamicSettings);
    }
}
