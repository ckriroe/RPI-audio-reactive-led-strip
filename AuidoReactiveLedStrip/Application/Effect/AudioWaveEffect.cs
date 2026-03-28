using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;
using Microsoft.Extensions.Options;

namespace Application.Effect
{
    public class AudioWaveEffect : AudioValueBasedEffect
    {
        private readonly IOptionsMonitor<StaticSettings> staticSettings;

        private float waveDistanceAccumulator = 0f;
        private LedPixel[] prevWaveStrip = [];

        public AudioWaveEffect(
            IAudioService audioService, 
            IOptionsMonitor<DynamicSettings> dynamicSettings,
            IOptionsMonitor<StaticSettings> staticSettings) 
            : base(audioService, dynamicSettings)
        {
            this.staticSettings = staticSettings;
        }

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            var dynamicSettings = base.dynamicSettings.CurrentValue;
            var staticSettings = this.staticSettings.CurrentValue;
            float newValue = base.GetCurrentAudioValue();

            int bounceLayers = dynamicSettings.BouncyWave ? staticSettings.BounceLayers : 0;

            int n = length * (1 + bounceLayers * 2);
            int center = dynamicSettings.EffectOrigin + length * bounceLayers;

            this.waveDistanceAccumulator += dynamicSettings.Speed / (float)dynamicSettings.Fps;
            int steps = (int)this.waveDistanceAccumulator;
            this.waveDistanceAccumulator -= steps;

            if (steps == 0 || center >= n)
            {
                if (prevStrip != null)
                    return prevStrip;
                else
                    return LedHelper.CreateEmptyStrip(length);
            }

            if (this.prevWaveStrip.Length != n)
                this.prevWaveStrip = LedHelper.CreateEmptyStrip(n).LedPixels;

            // TODO check if allocation can be removed

            LedPixel[] newStrip = this.prevWaveStrip
                .Select(p => new LedPixel(p.Value))
                .ToArray();

            // TODO: can these 2 be merged?
            MoveLeftSide(n, center, steps, newStrip);
            MoveRightSide(n, center, steps, newStrip);
            FillInterpolatedCenter(newValue, n, center, steps, newStrip);

            newStrip[center].Value = newValue;
            // TODO: check if this can be integrated into other loops
            FadeWaveStrip(newStrip, center, dynamicSettings);

            this.prevWaveStrip = newStrip;

            if (bounceLayers == 0)
                return new LedStrip { LedPixels = newStrip };
            else 
                return CreateBouncedWaveStrip(length, bounceLayers, newStrip);
        }

        private static LedStrip CreateBouncedWaveStrip(int length, int bounceLayers, LedPixel[] newStrip)
        {
            int bounceLayerLength = length;
            var bounced = LedHelper.CreateEmptyStrip(bounceLayerLength);

            for (int i = 0; i < bounced.LedPixels.Length; i++)
            {
                float baseVal = newStrip[i + bounceLayers * bounceLayerLength].Value;

                for (int bounceLayer = 0; bounceLayer < bounceLayers; bounceLayer++)
                {
                    bool isReverse = bounceLayer % 2 == 0;

                    int leftIndex = ((bounceLayers - 1) - bounceLayer) * bounceLayerLength;
                    if (isReverse)
                        leftIndex = (leftIndex + (bounceLayerLength - 1)) - i;
                    else
                        leftIndex += i;

                    int rightIndex = (bounceLayers + bounceLayer + 1) * bounceLayerLength;
                    if (isReverse)
                        rightIndex = (rightIndex + (bounceLayerLength - 1)) - i;
                    else
                        rightIndex += i;

                    float leftVal = newStrip[leftIndex].Value;
                    float rightVal = newStrip[rightIndex].Value;

                    if (leftVal > baseVal)
                        baseVal = leftVal;

                    if (rightVal > baseVal)
                        baseVal = rightVal;
                }

                bounced.LedPixels[i].Value = baseVal;
            }

            return bounced;
        }

        private void FadeWaveStrip(IList<LedPixel> stripToFade, int center, DynamicSettings settings)
        {
            for (int i = 0; i < stripToFade.Count; i++)
            {
                int distToCenter = Math.Abs(center - i);

                float factor = 1f - (distToCenter * (settings.FadeOverTime / 1000.0f));
                factor = Math.Clamp(factor, 0f, 10f);

                stripToFade[i].Value *= factor;
            }
        }

        private static void FillInterpolatedCenter(float newValue, int n, int center, int steps, LedPixel[] newStrip)
        {
            foreach (int direction in new[] { -1, 1 })
            {
                int idx = center + direction;

                if (idx < 0 || idx >= n)
                    continue;

                while (idx >= 0 && idx < n && newStrip[idx].Value == 0f)
                {
                    idx += direction;
                }

                if (idx < 0 || idx >= n)
                    continue;

                int targetIdx = idx;
                float targetValue = newStrip[targetIdx].Value;

                int gap = Math.Abs(targetIdx - center) - 1;
                int fill = Math.Min(gap, steps);

                for (int i = 1; i <= fill; i++)
                {
                    int pos = center + direction * i;
                    float t = (float)i / (gap + 1);
                    float v = MathHelper.Lerp(newValue, targetValue, t);
                    newStrip[pos].Value = v;
                }
            }
        }

        private void MoveRightSide(int n, int center, int steps, LedPixel[] newStrip)
        {
            for (int i = n - 1; i >= center; i--)
            {
                int srcIndex = i;
                int dstIndex = i + steps;

                if (dstIndex > n - 1 || srcIndex < 0)
                    continue;

                newStrip[dstIndex].Value = this.prevWaveStrip[srcIndex].Value;

                if (srcIndex != center)
                    newStrip[srcIndex].Value = 0f;
            }
        }

        private void MoveLeftSide(int n, int center, int steps, LedPixel[] newStrip)
        {
            for (int i = 0; i <= center; i++)
            {
                int srcIndex = i;
                int dstIndex = i - steps;

                if (dstIndex < 0 || srcIndex > n - 1)
                    continue;

                newStrip[dstIndex].Value = prevWaveStrip[srcIndex].Value;

                if (srcIndex != center)
                    newStrip[srcIndex].Value = 0f;
            }
        }
    }
}
