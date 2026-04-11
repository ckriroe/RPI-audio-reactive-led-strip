using Application.Audio.Service;
using Application.Domain;
using Application.Gpio;
using Application.RuntimeSettings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public class GpioExternalEffect : IStatefulEffect
    {
        private readonly GpioWrapper gpioWrapper;

        private StaticSettings? staticSettings = null;
        private bool isEnabled = false;
        private bool pendingGpioEnable = false;
        private bool isGpioTurnedOn = false;

        public GpioExternalEffect(IOptionsMonitor<StaticSettings> staticSettings, GpioWrapper gpioWrapper)
        {
            this.gpioWrapper = gpioWrapper;
        }

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => false;

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings)
        {
            this.staticSettings = staticSettings;
        }

        public void DisableEffect()
        {
            StaticSettings? staticSettings = this.staticSettings;

            if (!this.isEnabled || staticSettings == null)
                return;

            this.gpioWrapper.TemporarilyChangeGpio(true, staticSettings);
            this.isGpioTurnedOn = false;
            this.pendingGpioEnable = false;
            this.isEnabled = false;
        }

        public void EnableEffect()
        {
            if (this.isEnabled)
                return;

            this.isGpioTurnedOn = false;
            this.pendingGpioEnable = false;
            this.isEnabled = true;
        }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            StaticSettings? staticSettings = this.staticSettings;
            if (staticSettings == null)
                return null;

            if (this.isEnabled && this.pendingGpioEnable && !this.isGpioTurnedOn || this.gpioWrapper.UpdatePending(staticSettings))
            {
                this.gpioWrapper.TemporarilyChangeGpio(false, staticSettings);
                this.isGpioTurnedOn = true;
            }

            if (this.isEnabled && !this.isGpioTurnedOn)
            {
                // Clear strip and wait a cycle to disable strip then
                this.pendingGpioEnable = true;
                return LedHelper.CreateEmptyStrip(length);
            }

            return null;
        }
    }
}
