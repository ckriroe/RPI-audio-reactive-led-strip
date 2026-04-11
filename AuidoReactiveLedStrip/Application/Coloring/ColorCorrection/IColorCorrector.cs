using Application.RuntimeSettings;

namespace Application.Coloring.ColorCorrection
{
    public interface IColorCorrector
    {
        (float R, float G, float B) ColorCorrect((float R, float G, float B) colors, DynamicEffectSettings dynamicSettings);
    }
}
