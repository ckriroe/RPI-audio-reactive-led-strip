using Application.Audio.AudioService;
using Application.Domain;
using Application.Settings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public class AudioSepctrumEffect : IEffect
    {
        private readonly IAudioService audioService;
        private readonly IOptionsMonitor<DynamicSettings> dynamicSettings;

        public AudioSepctrumEffect(IAudioService audioService, IOptionsMonitor<DynamicSettings> dynamicSettings)
        {
            this.audioService = audioService;
            this.dynamicSettings = dynamicSettings;
        }

        public bool IsStatic => false;

        public bool UseAudioFft => true;

        public bool UseAudioValue => false;

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            var settings = this.dynamicSettings.CurrentValue;
            int center = settings.EffectOrigin;
            float[]? bins = this.audioService.GetCurrentFftData();
            int binCount = bins?.Length ?? 0;
            if (bins == null || binCount == 0)
                return prevStrip ?? LedHelper.CreateEmptyStrip(length);

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            int maxDist = Math.Max(center, (length - 1) - center);
            double sectionLength = binCount / (double)settings.SpectrumSections;

            float binMax = Math.Min(bins.Max(), settings.MaxFreqAmplitude);

            for (int i = 0; i < length; i++)
            {
                int dist = Math.Abs(i - center);

                float amp;

                if (binCount == 1)
                {
                    amp = bins[0];
                } 
                else if (settings.StaticSpectrum)
                {
                    double binPos = (dist / (double)maxDist) * (binCount - 1);
                    int section = (int)(binPos / sectionLength);

                    int startBin = Math.Max(0, (int)(section * binCount / (float)settings.SpectrumSections));
                    int endBin = Math.Min(binCount - 1, (int)((section + 1) * binCount / (float)settings.SpectrumSections));

                    if (startBin == endBin)
                    {
                        amp = bins[startBin];
                    } 
                    else
                    {
                        float maxVal = bins[startBin];
                        for (int b = startBin + 1; b <= endBin; b++)
                            if (bins[b] > maxVal) maxVal = bins[b];

                        amp = maxVal;
                    }
                }
                else
                {
                    double binPos = ((double)dist / maxDist) * (binCount - 1);
                    int b0 = (int)binPos;
                    int b1 = Math.Min(b0 + 1, binCount - 1);
                    double t = binPos - b0;

                    amp = (float)((1 - t) * bins[b0] + t * bins[b1]);
                }

                if (amp > settings.MaxFreqAmplitude)
                    amp = settings.MaxFreqAmplitude;

                if (amp < settings.MinFreqAmplitude)
                    amp = 0f;

                amp *= settings.ValueIncreaseFactor;

                float value;

                if (settings.MaxFreqAmplitude == 0f)
                {
                    value = 0f;
                } 
                else
                {
                    if (settings.AudioMode == AudioMode.Dynamic)
                        value = amp / binMax;
                    else
                        value = amp / settings.MaxFreqAmplitude;

                    value = Math.Clamp(value, 0f, 1f);
                }

                float prevValue = prevStrip.LedPixels[i].Value;

                if (value > prevValue && value > settings.SaturateThreshold)
                {
                    value = MathHelper.Lerp(prevValue, value, settings.Saturate);
                } 
                else
                {
                    value = prevValue * settings.Fade;
                }

                prevStrip.LedPixels[i].Value = value;
            }

            return prevStrip;
        }
    }
}
