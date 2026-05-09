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
using Application.LedStripRendering;
using Application.Looper;
using Application.RuntimeSettings;
using Application.Sequencing;
using Application.Visualization;
using Application.Visualization.Led;
using Application.Visualization.Screen;
using AudioProcessing.AudioProcessor;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

var builder = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((hostingContext, config) =>
    {
        config.AddJsonFile(
            "static_settings.json",
            optional: false,
            reloadOnChange: true
        );

        config.AddJsonFile(
            "presets.json",
            optional: false,
            reloadOnChange: true
        );
    })
    .ConfigureServices((context, services) =>
    {
        services.AddOptions<StaticSettings>()
            .Bind(context.Configuration)
            .PostConfigure(staticSetting => SettingsCorrector.CorrectStaticSettings(staticSetting));

        services.AddOptions<DynamicPresetSettings>()
            .Bind(context.Configuration)
            .PostConfigure<IOptionsMonitor<StaticSettings>>((dynamicPresetSettings, staticSettings) => SettingsCorrector.CorrectDynamicPresetSettings(dynamicPresetSettings, staticSettings.CurrentValue));

        services.AddSingleton<Application.Application.Lifetime.IApplicationLifetime, ApplicationLifetime>();
        services.AddSingleton<IApplicationService, ApplicationService>();
        services.AddSingleton<IVisualizerFactory, OpenTkScreenVisualizerFactory>();
        services.AddSingleton<ILooperFactory, StaticLooperFactory>();
        services.AddSingleton<Ws281xLedVisualizer>();
        services.AddSingleton<IGpioControllerFactory, GpioControllerFactory>();
        services.AddSingleton<GpioWrapper>();

        services.AddTransient<ILedStripSequencer, LedStripSequencer>();
        services.AddTransient<ILedStripRenderer, LedStripRenderer>();
        services.AddTransient<Accelerator>();
        services.AddTransient<Patternizer>();
        services.AddTransient<Repeater>();
        services.AddTransient<IRemapService, RemapService>();

        services.AddTransient<ValueColorMode>();
        services.AddTransient<IndexColorMode>();
        services.AddTransient<DistanceToCenterColorMode>();
        services.AddTransient<DistanceToBorderColorMode>();
        services.AddTransient<ColorWaveColorMode>();
        services.AddTransient<ColorIslandColorMode>();
        services.AddTransient<BrightnessAdjuster>();
        services.AddTransient<GammaCorrector>();
        services.AddTransient<INoiseGenerator, NoiseGenerator>();
        services.AddTransient<IValueSanitizer, ValueSanitizer>();
        services.AddTransient<IColorService, ColorService>();

        services.AddTransient<AudioLineDescendingEffect>();
        services.AddTransient<AudioLineEffect>();
        services.AddTransient<AudioPulseEffect>();
        services.AddTransient<AudioRandomBurstEffect>();
        services.AddTransient<AudioSepctrumEffect>();
        services.AddTransient<AudioWaveEffect>();
        services.AddTransient<GpioExternalEffect>();
        services.AddTransient<StaticAscendingValueEffect>();
        services.AddTransient<StaticValueOneEffect>();
        services.AddTransient<IEffectService, EffectService>();

        services.AddTransient<AudioFftDataProvider>();
        services.AddTransient<SimpleAudioValueProvider>();
        services.AddTransient<MovingMaxAudioValueProvider>();
        services.AddTransient<TransientAudioValueProvider>();
        
        services.AddTransient<IAudioFftTransformer, AudioFftTransformer>();
        services.AddTransient<IAudioService, AudioService>();
        services.AddSingleton<IAudioReceiver, PortAudioReceiver>();
        services.AddSingleton<IOrchestrator, Orchestrator>();
    })
    .Build();

IApplicationService applicationService = builder.Services.GetRequiredService<IApplicationService>();
applicationService.StartApplication();
