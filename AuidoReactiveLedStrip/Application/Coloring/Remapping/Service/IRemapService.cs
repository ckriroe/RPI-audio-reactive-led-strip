using Application.Domain;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Remapping.Service
{
    public interface IRemapService
    {
        Color[]? RemapColors(LedStrip ledStrip);

        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);
    }
}