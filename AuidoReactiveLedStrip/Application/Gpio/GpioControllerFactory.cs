using System.Device.Gpio;

namespace Application.Gpio
{
    public class GpioControllerFactory : IGpioControllerFactory
    {
        private GpioController? gpioController = null;

        public GpioController? GetGpioController()
        {
            if (OperatingSystem.IsLinux() && this.gpioController == null)
            {
                this.gpioController = new GpioController();
            }

            return this.gpioController;
        }
    }
}
