using Application.Settings;

namespace Application.Coloring.ColorCorrection
{
    public class GammaCorrector : IColorCorrector
    {
        public (float R, float G, float B) ColorCorrect((float R, float G, float B) colors, DynamicSettings dynamicSettings)
        {
            return (
                R: this.GammaCorrect(colors.R, dynamicSettings),
                G: this.GammaCorrect(colors.G, dynamicSettings),
                B: this.GammaCorrect(colors.B, dynamicSettings)
            );
        }

        private float GammaCorrect(float value, DynamicSettings dynamicSettings)
        {
            return (float)Math.Pow(value / 255.0f, dynamicSettings.Gamma) * 255.0f + 0.5f;
        }
    }
}
