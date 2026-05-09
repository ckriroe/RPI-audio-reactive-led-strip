using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioSepctrumEffect : IEffect
    {
        private IAudioService? audioService;
        private DynamicEffectSettings? dynamicEffectSettings;

        public AudioSepctrumEffect(IAudioService audioService)
        {
            this.audioService = audioService;
        }

        public bool IsStatic => false;

        public bool UseAudioFft => true;

        public bool UseAudioValue => false;

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings)
        {
            this.audioService = audioService;
            this.dynamicEffectSettings = dynamicEffectSettings;
        }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? settings = this.dynamicEffectSettings;
            if (settings == null)
                return null;

            int center = settings.EffectOrigin;
            float[]? bins = this.audioService?.GetCurrentFftData();
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
                    if (settings.AudioMode == AudioMode.Static)
                        value = amp / settings.MaxFreqAmplitude;
                    else
                        value = amp / binMax;

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
