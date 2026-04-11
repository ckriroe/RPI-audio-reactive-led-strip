using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public interface IColorMode
    {
        void PrecomputeValues(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings, LedStrip ledStrip);

        Color GetColorForValue(DynamicEffectSettings dynamicSettings, float audioValue, int index, int length);
    }
}
