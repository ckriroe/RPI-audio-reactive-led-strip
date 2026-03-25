using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Mode
{
    public interface IColorMode
    {
        void PrecomputeValues(StaticSettings staticSettings, DynamicSettings dynamicSettings, LedStrip ledStrip);

        Color GetColorForValue(DynamicSettings dynamicSettings, float audioValue, int index, int length);
    }
}
