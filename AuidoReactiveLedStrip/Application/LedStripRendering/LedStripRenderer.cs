using Application.Audio.Service;
using Application.Coloring.Remapping.Service;
using Application.Coloring.Service;
using Application.Domain;
using Application.Effect.Service;
using Application.RuntimeSettings;
using System.Drawing;

namespace Application.LedStripRendering
{
    public class LedStripRenderer : ILedStripRenderer
    {
        private readonly IAudioService audioService;
        private readonly IEffectService effectService;
        private readonly IColorService colorService;
        private readonly IRemapService remapService;

        public LedStripRenderer(
            IAudioService audioService,
            IEffectService effectService,
            IColorService colorService,
            IRemapService remapService)
        {
            this.audioService = audioService;
            this.effectService = effectService;
            this.colorService = colorService;
            this.remapService = remapService;
        }

        public Color[]? RenderLedStrip()
        {
            LedStrip? ledStrip = effectService.GetLedStrip();

            if (ledStrip != null)
            {
                this.colorService.ColorizeLedStrip(ledStrip);
                return this.remapService.RemapColors(ledStrip);
            }
            else
            {
                return null;
            }
        }

        public void ApplySettings(StaticSettings staticSettings, DynamicEffectSettings dynamicSettings)
        {
            this.audioService.ApplySettings(staticSettings, dynamicSettings);
            this.effectService.ApplySettings(this.audioService, staticSettings, dynamicSettings);
            this.colorService.ApplySettings(staticSettings, dynamicSettings);
            this.remapService.ApplySettings(staticSettings, dynamicSettings);
        }

        public void Disable()
        {
            this.audioService.SetAudioMode(AudioServiceMode.None);
        }
    }
}
