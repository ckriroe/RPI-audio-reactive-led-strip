using Application.Domain;
using Application.RuntimeSettings;

namespace Application.Coloring.Service
{
    public interface IColorService
    {
        void ColorizeLedStrip(LedStrip ledStrip);
        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);
    }
}
