using Application.RuntimeSettings;
using System.Device.Gpio;

namespace Application.Gpio
{
    public class GpioWrapper
    {
        private readonly GpioController? gpioController;

        private bool isGloballyEnabled = true;
        private bool isTemporarilyEnabled = true;
        private bool currentOutputWhenGpioOff;
        private int? externalGpioPin;

        public GpioWrapper(IGpioControllerFactory gpioControllerFactory)
        {
            this.gpioController = gpioControllerFactory.GetGpioController();
        }

        public void GloballyChangeGpio(bool enabled, StaticSettings staticSettings)
        {
            bool wasEnabled = this.IsCurrentlyEnabled();
            this.isGloballyEnabled = enabled;
            bool isEnabled = this.IsCurrentlyEnabled();
            this.CheckForGpioStateChange(staticSettings, wasEnabled, isEnabled);
        }

        public void TemporarilyChangeGpio(bool enabled, StaticSettings staticSettings)
        {
            bool wasEnabled = this.IsCurrentlyEnabled();
            this.isTemporarilyEnabled = enabled;
            bool isEnabled = this.IsCurrentlyEnabled();
            this.CheckForGpioStateChange(staticSettings, wasEnabled, isEnabled);
        }

        private void CheckForGpioStateChange(StaticSettings staticSettings, bool wasEnabled, bool isEnabled)
        {
            if (wasEnabled != isEnabled || this.currentOutputWhenGpioOff != staticSettings.OutputWhenGpioOff || this.externalGpioPin != staticSettings.ExternalModeRelayGpio)
            {
                this.ChangeGpioState(isEnabled, staticSettings);
            }
        }

        private void ChangeGpioState(bool isEnabled, StaticSettings staticSettings)
        {
            if (!OperatingSystem.IsLinux() || this.gpioController == null)
                return;

            if (this.externalGpioPin == null || staticSettings.ExternalModeRelayGpio != this.externalGpioPin)
            {
                if (this.externalGpioPin != null && this.gpioController.IsPinOpen(this.externalGpioPin.Value))
                {
                    this.gpioController.Write(this.externalGpioPin.Value, this.GetPinValue(false));
                    this.gpioController.ClosePin(this.externalGpioPin.Value);
                }

                if (!this.gpioController.IsPinOpen(staticSettings.ExternalModeRelayGpio))
                    this.gpioController.OpenPin(staticSettings.ExternalModeRelayGpio, PinMode.Output);

                this.externalGpioPin = staticSettings.ExternalModeRelayGpio;
            }

            if (isEnabled)
            {
                this.currentOutputWhenGpioOff = staticSettings.OutputWhenGpioOff;
                this.gpioController.Write(this.externalGpioPin.Value, this.GetPinValue(true));
            }
            else
            {
                this.gpioController.Write(this.externalGpioPin.Value, this.GetPinValue(false));
                this.currentOutputWhenGpioOff = staticSettings.OutputWhenGpioOff;
            }
        }

        private PinValue GetPinValue(bool enabled)
        {
            if (this.currentOutputWhenGpioOff)
            {
                return enabled ? PinValue.Low : PinValue.High;
            }
            else
            {
                return enabled ? PinValue.High : PinValue.Low;
            }
        }

        private bool IsCurrentlyEnabled()
        {
            return this.isGloballyEnabled && this.isTemporarilyEnabled;
        }

        public bool UpdatePending(StaticSettings currentStaticSettings)
        {
            return this.externalGpioPin != currentStaticSettings.ExternalModeRelayGpio ||
                this.currentOutputWhenGpioOff != currentStaticSettings.OutputWhenGpioOff;
        }
    }
}
