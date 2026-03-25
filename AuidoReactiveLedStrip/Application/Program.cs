using Application.Application.Lifetime;
using Application.Application.Orchestration;
using Application.Application.Service;
using Application.Audio.FftTransformer;
using Application.Audio.Receiver;
using Application.Audio.Service;
using Application.Audio.ValueProvider;
using Application.Coloring.ColorCorrection;
using Application.Coloring.Mode;
using Application.Coloring.Noise;
using Application.Coloring.Remapping;
using Application.Coloring.Remapping.Service;
using Application.Coloring.Sanitizing;
using Application.Coloring.Service;
using Application.Effect;
using Application.Effect.Service;
using Application.Gpio;
using Application.Looper;
using Application.RuntimeSettings;
using Application.Visualization;
using Application.Visualization.Led;
using Application.Visualization.Screen;
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
        services.AddOptions<StaticSettings>()
            .Bind(context.Configuration)
            .PostConfigure(o => SettingsCorrector.CorrectStaticSettings(o));

        services.AddOptions<DynamicSettings>()
            .Bind(context.Configuration)
            .PostConfigure(o => SettingsCorrector.CorrectDynamicSettings(o));

        services.AddSingleton<Application.Application.Lifetime.IApplicationLifetime, ApplicationLifetime>();
        services.AddSingleton<IApplicationService, ApplicationService>();
        services.AddSingleton<IVisualizerFactory, OpenTkScreenVisualizerFactory>();

        if (OperatingSystem.IsWindows())
            services.AddTransient<ILooper, WindowsOverheadLooper>();
        else
            services.AddTransient<ILooper, LinuxLooper>();

        services.AddSingleton<Accelerator>();
        services.AddSingleton<Patternizer>();
        services.AddSingleton<Repeater>();
        services.AddSingleton<IRemapService, RemapService>();

        services.AddSingleton<Ws281xLedVisualizer>();
        services.AddSingleton<ValueColorMode>();
        services.AddSingleton<IndexColorMode>();
        services.AddSingleton<DistanceToCenterColorMode>();
        services.AddSingleton<DistanceToBorderColorMode>();
        services.AddSingleton<ColorWaveColorMode>();
        services.AddSingleton<ColorIslandColorMode>();
        services.AddSingleton<BrightnessAdjuster>();
        services.AddSingleton<GammaCorrector>();
        services.AddSingleton<INoiseGenerator, NoiseGenerator>();
        services.AddSingleton<IValueSanitizer, ValueSanitizer>();
        services.AddSingleton<IColorService, ColorService>();

        services.AddSingleton<IGpioControllerFactory, GpioControllerFactory>();
        services.AddSingleton<AudioLineDescendingEffect>();
        services.AddSingleton<AudioLineEffect>();
        services.AddSingleton<AudioPulseEffect>();
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
        services.AddSingleton<IOrchestrator, Orchestrator>();
    })
    .Build();

IApplicationService applicationService = builder.Services.GetRequiredService<IApplicationService>();
applicationService.StartApplication();


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