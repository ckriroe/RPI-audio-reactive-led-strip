using Application.RuntimeSettings;
using System.Drawing;

namespace Application.LedStripRendering
{
    public interface ILedStripRenderer
    {
        Color[]? RenderLedStrip();

        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);

        void Disable();

        void Reset();
    }
}
