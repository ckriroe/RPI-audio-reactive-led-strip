using Application.Domain;
using Application.Gpio;
using Application.RuntimeSettings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public class GpioExternalEffect : IStatefulEffect
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;
        private readonly GpioWrapper gpioWrapper;

        private bool isEnabled = false;
        private bool pendingGpioEnable = false;
        private bool isGpioTurnedOn = false;

        public GpioExternalEffect(IOptionsMonitor<StaticSettings> staticSettings, GpioWrapper gpioWrapper)
        {
            this.staticSettings = staticSettings;
            this.gpioWrapper = gpioWrapper;
        }

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => false;

        public void DisableEffect()
        {
            if (!this.isEnabled)
                return;

            this.gpioWrapper.TemporarilChangeGpio(true, this.staticSettings.CurrentValue);
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
            StaticSettings currentStaticSettings = this.staticSettings.CurrentValue;

            if (this.isEnabled && this.pendingGpioEnable && !this.isGpioTurnedOn || this.gpioWrapper.UpdatePending(currentStaticSettings))
            {
                this.gpioWrapper.TemporarilChangeGpio(false, currentStaticSettings);
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
