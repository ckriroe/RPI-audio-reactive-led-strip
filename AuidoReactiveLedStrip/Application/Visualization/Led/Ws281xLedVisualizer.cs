using Application.Gpio;
using Application.RuntimeSettings;
using Microsoft.Extensions.Options;
using rpi_ws281x;
using System.Drawing;

namespace Application.Visualization.Led
{
    public class Ws281xLedVisualizer : IVisualizer
    {
        private readonly IOptionsMonitor<DynamicSettings> dynamicSettings;
        private readonly IOptionsMonitor<StaticSettings> staticSettings;
        private readonly GpioWrapper gpioWrapper;

        private bool isActive = false;
        private int currentUpdateFrequency;
        private Settings? settings = null;
        private Controller? controller = null;
        private WS281x? ws281x = null;        

        public Ws281xLedVisualizer(
            IOptionsMonitor<DynamicSettings> dynamicSettings,
            IOptionsMonitor<StaticSettings> staticSettings,
            GpioWrapper gpioWrapper
        )
        {
            this.dynamicSettings = dynamicSettings;
            this.staticSettings = staticSettings;
            this.gpioWrapper = gpioWrapper;
        }

        public void Start()
        {
            if (this.isActive)
                return;

            this.InitLedStrip(this.dynamicSettings.CurrentValue.LedCount);
        }

        private void InitLedStrip(int pixels) 
        {
            if (OperatingSystem.IsLinux())
            {
                StaticSettings currentStaticSettings = this.staticSettings.CurrentValue;
                this.gpioWrapper.GloballyChangeGpio(true, currentStaticSettings);
                int ledUpdateFrequency = currentStaticSettings.LedUpdateFrequency;
                this.settings = new Settings((uint)ledUpdateFrequency, Settings.DEFAULT_DMA_CHANNEL);
                this.controller = this.settings.AddController(pixels, Pin.Gpio12, StripType.WS2811_STRIP_GRB, ControllerType.Unknown, 255, false);
                this.ws281x = new WS281x(this.settings);
                this.isActive = true;
                this.currentUpdateFrequency = ledUpdateFrequency;
            }
        }

        public void Stop()
        {
            if (!this.isActive)
                return;

            this.controller?.Reset();
            this.ws281x?.Render();
            this.gpioWrapper.GloballyChangeGpio(false, this.staticSettings.CurrentValue);
            this.isActive = false;
        }

        public void UpdateColors(Color[] colors)
        {
            if (!this.isActive || this.controller == null || this.ws281x == null)
                return;

            StaticSettings currentStaticSettings = this.staticSettings.CurrentValue;
            if (colors.Length != this.controller.LEDCount || this.currentUpdateFrequency != currentStaticSettings.LedUpdateFrequency || this.gpioWrapper.UpdatePending(currentStaticSettings))
            {
                this.Dispose();
                this.InitLedStrip(colors.Length);
            }

            for (int i = 0; i < colors.Length; i++)
            {
                this.controller.SetLED(i, colors[i]);
            }

            this.ws281x?.Render();
        }

        public void Dispose()
        {
            this.Stop();
            this.ws281x?.Dispose();

            this.controller = null;
            this.ws281x = null;
            this.settings = null;
        }
    }
}
