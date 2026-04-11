using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Sequencing
{
    public interface ILedStripSequencer
    {
        Color[]? RenderLedStrip();

        void ApplySettings(StaticSettings staticSettings, DynamicPresetSettings dynamicPresetSettings);

        void Disable();
    }
}