using Application.Audio.Service;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class FireSimulationEffect : IEffect, IStatefulEffect
    {
        private const float TimeScale = 2.5f;
        private const float BaseSpeed = 60.0f;
        private const float MinFlameScale = 0.40f;
        private const float MaxFlameScale = 0.98f;
        private const float ParticleSizeBase = 30.0f;
        private const float MinParticleSize = 10.0f;
        private const float MinSpatialScale = 0.5f;
        private const float MaxSpatialScale = 2.0f;

        private const int MaxEmbersCount = 5;
        private const double EmberSpawnChance = 0.12;
        private const float EmberMinDuration = 0.5f;
        private const float EmberDurationRange = 0.8f;
        private const float EmberMinIntensity = 0.45f;
        private const float EmberIntensityRange = 0.35f;
        private const float EmberMinRadius = 2.0f;
        private const float EmberRadiusRange = 2.5f;
        private const float EmberAttackRatio = 0.3f;
        private const float EmberDecayRatio = 0.7f;
        private const float EmberInfluenceRadiusMultiplier = 2.5f;

        private const float Octave1SpatialFrequency = 0.022f;
        private const float Octave1TimeSpeed = 0.45f;

        private const float Octave2SpatialFrequency = 0.058f;
        private const float Octave2TimeSpeed = 1.10f;
        private const float Octave2Weight = 0.50f;

        private const float Octave3SpatialFrequency = 0.145f;
        private const float Octave3TimeSpeed = 2.60f;
        private const float Octave3Weight = 0.25f;

        private const float TotalNoiseWeight = 1.75f;

        private const float CoalBedIntensity = 0.12f;
        private const float CoalBedSpatialFrequency = 0.015f;
        private const float CoalBedTimeSpeed = 0.15f;

        private const float FlameScaleCutoffMultiplier = 0.9f;
        private const float MinCutoff = 0.10f;
        private const float MaxCutoff = 0.60f;
        private const float FlameContrastPower = 1.6f;
        private const float FlameIntensityMultiplier = 1.15f;

        private const float HashFrequencyMultiplier = 127.1f;
        private const float HashAmplitudeMultiplier = 43758.5453f;

        private class Ember
        {
            public float Position;
            public float Life;
            public float Duration;
            public float MaxIntensity;
            public float Radius;
        }

        private float time = 0f;
        private readonly List<Ember> embers = new();
        private DynamicEffectSettings? dynamicSettings = null;
        private StaticSettings? staticSettings = null;

        public bool IsStatic => false;
        public bool UseAudioFft => false;
        public bool UseAudioValue => false;

        public void ApplySettings(IAudioService audioService, StaticSettings staticSettings, DynamicEffectSettings dynamicEffectSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicSettings = dynamicEffectSettings;
        }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = this.dynamicSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            int framesPerSecond = Math.Max(1, staticSettings.Fps);
            float deltaTime = 1.0f / framesPerSecond;

            float speed = Math.Max(1.0f, (float)dynamicSettings.Speed);
            this.time += (speed / BaseSpeed) * deltaTime * TimeScale;

            float flameScale = Math.Clamp(dynamicSettings.Fade, MinFlameScale, MaxFlameScale);
            float spatialScale = Math.Clamp(ParticleSizeBase / Math.Max(MinParticleSize, (float)dynamicSettings.ParticleSize), MinSpatialScale, MaxSpatialScale);

            for (int emberIndex = this.embers.Count - 1; emberIndex >= 0; emberIndex--)
            {
                var ember = this.embers[emberIndex];
                ember.Life += deltaTime / ember.Duration;
                if (ember.Life >= 1.0f)
                {
                    this.embers.RemoveAt(emberIndex);
                }
            }

            if (this.embers.Count < MaxEmbersCount && Random.Shared.NextDouble() < EmberSpawnChance)
            {
                this.embers.Add(new Ember
                {
                    Position = Random.Shared.Next(0, length),
                    Life = 0f,
                    Duration = EmberMinDuration + (float)Random.Shared.NextDouble() * EmberDurationRange,
                    MaxIntensity = EmberMinIntensity + (float)Random.Shared.NextDouble() * EmberIntensityRange,
                    Radius = EmberMinRadius + (float)Random.Shared.NextDouble() * EmberRadiusRange
                });
            }

            for (int i = 0; i < length; i++)
            {
                float samplePosition = i * spatialScale;

                float macroNoise = Noise1D(samplePosition * Octave1SpatialFrequency + this.time * Octave1TimeSpeed);
                float driftNoise = Noise1D(samplePosition * Octave2SpatialFrequency - this.time * Octave2TimeSpeed) * Octave2Weight;
                float crackleNoise = Noise1D(samplePosition * Octave3SpatialFrequency + this.time * Octave3TimeSpeed) * Octave3Weight;

                float totalNoise = (macroNoise + driftNoise + crackleNoise) / TotalNoiseWeight;
                float coalBed = CoalBedIntensity * Noise1D(samplePosition * CoalBedSpatialFrequency + this.time * CoalBedTimeSpeed);

                float cutoff = Math.Clamp(1.0f - flameScale * FlameScaleCutoffMultiplier, MinCutoff, MaxCutoff);
                float flameIntensity = MathF.Max(0.0f, (totalNoise - cutoff) / (1.0f - cutoff));
                flameIntensity = MathF.Pow(flameIntensity, FlameContrastPower);

                float pixelValue = coalBed + (flameIntensity * flameScale * FlameIntensityMultiplier);

                foreach (var ember in this.embers)
                {
                    float distanceToEmber = MathF.Abs(i - ember.Position);
                    if (distanceToEmber < ember.Radius * EmberInfluenceRadiusMultiplier)
                    {
                        float lifeFactor = ember.Life < EmberAttackRatio
                            ? (ember.Life / EmberAttackRatio)
                            : ((1.0f - ember.Life) / EmberDecayRatio);

                        float spatialFactor = MathF.Exp(-(distanceToEmber * distanceToEmber) / (2.0f * ember.Radius * ember.Radius));
                        pixelValue += ember.MaxIntensity * lifeFactor * spatialFactor;
                    }
                }

                prevStrip.LedPixels[i].Value = MathHelper.Clamp(pixelValue, 0.0f, 1.0f);
            }

            return prevStrip;
        }

        private static float Noise1D(float samplePosition)
        {
            float cellIndex = MathF.Floor(samplePosition);
            float fractionalOffset = samplePosition - cellIndex;
            float smoothWeight = fractionalOffset * fractionalOffset * (3.0f - 2.0f * fractionalOffset);

            float hashLeft = Hash(cellIndex);
            float hashRight = Hash(cellIndex + 1.0f);

            return (1.0f - smoothWeight) * hashLeft + smoothWeight * hashRight;
        }

        private static float Hash(float value)
        {
            float sin = MathF.Sin(value * HashFrequencyMultiplier) * HashAmplitudeMultiplier;
            return sin - MathF.Floor(sin);
        }

        public void EnableEffect()
        {
            this.embers.Clear();
            this.time = 0f;
        }

        public void DisableEffect()
        {
            this.embers.Clear();
            this.time = 0f;
        }
    }
}
