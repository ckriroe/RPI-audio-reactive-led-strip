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
        services.AddSingleton<GpioWrapper>();
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
