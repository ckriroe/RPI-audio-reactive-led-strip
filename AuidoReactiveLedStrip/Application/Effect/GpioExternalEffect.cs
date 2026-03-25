using Application.Domain;
using Application.Gpio;
using Application.RuntimeSettings;
using Application.Util;
using Microsoft.Extensions.Options;
using System.Device.Gpio;
using System.Security.AccessControl;

namespace Application.Effect
{
    public class GpioExternalEffect : IStatefulEffect
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;
        private readonly IGpioControllerFactory gpioControllerFactory;
        private readonly GpioController? gpioController = null;

        private bool isEnabled = false;
        private bool pendingGpioEnable = false;
        private bool isGpioTurnedOn = false;
        private int? lastGpioPin = null;

        public GpioExternalEffect(IOptionsMonitor<StaticSettings> staticSettings, IGpioControllerFactory gpioControllerFactory)
        {
            this.staticSettings = staticSettings;
            this.gpioControllerFactory = gpioControllerFactory;
            this.gpioController = this.gpioControllerFactory.GetGpioController();
        }

        public bool IsStatic => false;

        public bool UseAudioFft => false;

        public bool UseAudioValue => false;

        public void DisableEffect()
        {
            if (!this.isEnabled)
                return;

            if (this.lastGpioPin != null)
            {
                if (this.isGpioTurnedOn)
                    this.gpioController?.Write(this.lastGpioPin.Value, PinValue.Low);

                this.gpioController?.ClosePin(this.lastGpioPin.Value);
            }

            this.isGpioTurnedOn = false;
            this.pendingGpioEnable = false;
            this.isEnabled = false;
            this.lastGpioPin = null;
        }

        public void EnableEffect()
        {
            if (this.isEnabled)
                return;

            this.lastGpioPin = this.staticSettings.CurrentValue.ExternalModeRelayGpio;
            this.gpioController?.OpenPin(this.lastGpioPin.Value, PinMode.Output);

            this.isGpioTurnedOn = false;
            this.pendingGpioEnable = false;
            this.isEnabled = true;
        }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            if (this.isEnabled && this.pendingGpioEnable && !this.isGpioTurnedOn && this.lastGpioPin != null)
            {
                this.gpioController?.Write(this.lastGpioPin.Value, PinValue.High);
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
