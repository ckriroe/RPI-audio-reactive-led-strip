using Application;
using Application.Audio.AudioReceiver;
using Application.Audio.AudioService;
using Application.Audio.AudioValueProvider;
using Application.Audio.FftTransformer;
using Application.Effect;
using Application.Effect.Service;
using Application.Gpio;
using Application.Looper;
using Application.Settings;
using AudioProcessing.AudioProcessor;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((hostingContext, config) =>
    {
        config.AddJsonFile(
            "static_settings.json",
            optional: false,
            reloadOnChange: true
        );

        config.AddJsonFile(
            "dynamic_settings.json",
            optional: false,
            reloadOnChange: true
        );
    })
    .ConfigureServices((context, services) =>
    {
        services.Configure<StaticSettings>(context.Configuration);
        services.Configure<DynamicSettings>(context.Configuration);

        if (OperatingSystem.IsWindows())
            services.AddTransient<ILooper, WindowsOverheadLooper>();
        else
            services.AddTransient<ILooper, LinuxLooper>();

        services.AddSingleton<IGpioControllerFactory, GpioControllerFactory>();
        services.AddSingleton<AudioLineEffekt>();
        services.AddSingleton<AudioPulseEffekt>();
        services.AddSingleton<AudioRandomBurstEffect>();
        services.AddSingleton<AudioSepctrumEffect>();
        services.AddSingleton<AudioWaveEffect>();
        services.AddSingleton<GpioExternalEffect>();
        services.AddSingleton<StaticAscendingValueEffect>();
        services.AddSingleton<StaticValueOneEffect>();
        services.AddSingleton<IEffectService, EffectService>();

        services.AddSingleton<AudioFftDataProvider>();
        services.AddSingleton<SimpleAudioValueProvider>();
        services.AddSingleton<MovingMaxAudioValueProvider>();
        
        services.AddSingleton<IAudioReceiver, PortAudioReceiver>();
        services.AddSingleton<IAudioFftTransformer, AudioFftTransformer>();
        services.AddSingleton<IAudioService, AudioService>();
        services.AddSingleton<Orchestrator>();
    })
    .Build();

Orchestrator processor = builder.Services.GetRequiredService<Orchestrator>();
processor.Start();
Console.WriteLine("Press enter to exit the application ...");
Console.Read();
processor.Stop();

/*using System.Drawing;

class Program
{
    static void Main()
    {
        /*
        //The default settings uses a frequency of 800000 Hz and the DMA channel 10.
        var settings = Settings.CreateDefaultSettings();

        //Use 16 LEDs and GPIO Pin 18.
        //Set brightness to maximum (255)
        //Use Unknown as strip type. Then the type will be set in the native assembly.
        var controller = settings.AddController(600, Pin.Gpio12, StripType.WS2811_STRIP_GRB, ControllerType.PCM, 255, false);

        using (var rpi = new WS281x(settings))
        {
            int x = 1;
            for (; ;)
            {
                Stopwatch sw = new Stopwatch();
                for (int i = 0; i < controller.LEDCount; i++)
                {
                    controller.SetLED(i, ColorFromHSV((10 * x + i) % 360, 1.0, 1.0));
                }
              
                sw.Start();
                rpi.Render();
                sw.Stop();
                Console.WriteLine(x + " TIME: " + sw.Elapsed.Milliseconds.ToString() + "ms");
                x++;
            }
        }
    }

    public static Color ColorFromHSV(double hue, double saturation, double value)
    {
        // hue: 0-360
        // saturation, value: 0-1
        int hi = (int)(hue / 60) % 6;
        double f = hue / 60 - Math.Floor(hue / 60);

        value = value * 255;
        int v = (int)value;
        int p = (int)(value * (1 - saturation));
        int q = (int)(value * (1 - f * saturation));
        int t = (int)(value * (1 - (1 - f) * saturation));

        return hi switch
        {
            0 => Color.FromArgb(v, t, p),
            1 => Color.FromArgb(q, v, p),
            2 => Color.FromArgb(p, v, t),
            3 => Color.FromArgb(p, q, v),
            4 => Color.FromArgb(t, p, v),
            5 => Color.FromArgb(v, p, q),
            _ => Color.FromArgb(0, 0, 0)
        };
    }
}*/