using Application.RuntimeSettings;
using System.Drawing;

namespace Application.LedStripRendering
{
    public interface ILedStripRenderer : IDisposable
    {
        Color[]? RenderLedStrip();

        void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings);
    }
}
