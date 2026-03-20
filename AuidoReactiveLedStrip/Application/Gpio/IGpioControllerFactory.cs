using System.Device.Gpio;

namespace Application.Gpio
{
    public interface IGpioControllerFactory
    {
        GpioController? GetGpioController();
    }
}