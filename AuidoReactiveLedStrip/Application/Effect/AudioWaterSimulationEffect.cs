using System;
using Application.Domain;
using Application.RuntimeSettings;
using Application.Util;

namespace Application.Effect
{
    public class AudioWaterSimulationEffect : AudioValueBasedEffect, IStatefulEffect
    {
        private const float AttackDeltaThreshold = 0.05f;
        private const float AttackAudioThreshold = 0.10f;
        private const float RhythmicAudioThreshold = 0.22f;
        private const int RhythmicDropIntervalMs = 120;
        private const int MinSplashIntervalMs = 75;

        private const int DefaultMaxWaves = 4;
        private const float HighIntensityThreshold = 0.70f;
        private const float MediumIntensityThreshold = 0.40f;
        private const float MediumIntensityWaveRatio = 0.60f;

        private const int MinBaseRadius = 2;
        private const int ParticleSizeDivisor = 4;
        private const float BaseForceMultiplier = 0.8f;
        private const float ForceMultiplierRandomRange = 0.5f;
        private const float MinSplashForce = 0.35f;
        private const float MaxSplashForce = 1.4f;
        private const float SplashVelocityFactor = 0.25f;

        private const float WavePropagationSpeedSquared = 0.28f;
        private const float MinFade = 0.001f;
        private const float MaxFade = 0.999f;
        private const float VelocityDampingScale = 0.18f;
        private const float VelocityDampingBase = 0.002f;
        private const float HeightDecayScale = 0.05f;
        private const float RestoringForceScale = 0.006f;
        private const float MinRestoringFade = 0.15f;

        private const float SpeedSubStepDivisor = 75.0f;
        private const int MinSubSteps = 1;
        private const int MaxSubSteps = 8;
        private const float BoundaryDampingFactor = 0.82f;

        private const float CrestWeight = 1.6f;
        private const float TroughWeight = 0.15f;
        private const float SlopeWeight = 0.35f;
        private const float MinCutoff = 0.015f;
        private const float CutoffFadeMultiplier = 0.035f;
        private const float MaxCutoff = 0.050f;

        private const float SettleTimeoutFadeMultiplier = 500f;
        private const float SettleTimeoutBaseMs = 120f;
        private const float SettleMaxActivityThreshold = 0.01f;

        private float[] heights = [];
        private float[] velocities = [];
        private float[] laplacians = [];

        private long lastSplashTime = 0;
        private float lastAudioValue = 0f;

        public override LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            DynamicEffectSettings? dynamicSettings = base.dynamicEffectSettings;
            StaticSettings? staticSettings = this.staticSettings;
            if (dynamicSettings == null || staticSettings == null)
                return null;

            prevStrip ??= LedHelper.CreateEmptyStrip(length);
            if (prevStrip.LedPixels.Length != length)
                prevStrip = LedHelper.CreateEmptyStrip(length);

            if (this.heights.Length != length)
            {
                this.heights = new float[length];
                this.velocities = new float[length];
                this.laplacians = new float[length];
            }

            float audioValue = base.GetCurrentAudioValue();
            long currentTicksMs = Environment.TickCount;

            float audioDelta = audioValue - this.lastAudioValue;
            this.lastAudioValue = audioValue;

            bool isBeatAttack = audioDelta > AttackDeltaThreshold && audioValue > AttackAudioThreshold;
            bool isRhythmicDrop = audioValue > RhythmicAudioThreshold && (currentTicksMs - this.lastSplashTime > RhythmicDropIntervalMs);

            if ((isBeatAttack || isRhythmicDrop) && (currentTicksMs - this.lastSplashTime > MinSplashIntervalMs))
            {
                this.lastSplashTime = currentTicksMs;

                int maxWaveCount = dynamicSettings.MaxWaves > 0 ? dynamicSettings.MaxWaves : DefaultMaxWaves;
                int dropCount;
                if (audioValue > HighIntensityThreshold)
                    dropCount = Math.Max(1, maxWaveCount);
                else if (audioValue > MediumIntensityThreshold)
                    dropCount = Math.Max(1, (int)MathF.Ceiling(maxWaveCount * MediumIntensityWaveRatio));
                else
                    dropCount = 1;

                int baseRadius = Math.Max(MinBaseRadius, dynamicSettings.ParticleSize / ParticleSizeDivisor);
                int segmentWidth = Math.Max(1, length / dropCount);

                for (int dropIndex = 0; dropIndex < dropCount; dropIndex++)
                {
                    int minX = dropIndex * segmentWidth;
                    int maxX = (dropIndex == dropCount - 1) ? length : (dropIndex + 1) * segmentWidth;
                    int dropPosition = (minX < maxX) ? Random.Shared.Next(minX, maxX) : Math.Clamp(minX, 0, length - 1);

                    int splashRadius = Math.Max(MinBaseRadius, baseRadius + Random.Shared.Next(-1, 2));
                    float forceMultiplier = BaseForceMultiplier + ((float)Random.Shared.NextDouble() * ForceMultiplierRandomRange);
                    float splashForce = Math.Clamp(audioValue * forceMultiplier, MinSplashForce, MaxSplashForce);

                    for (int radiusOffset = -splashRadius; radiusOffset <= splashRadius; radiusOffset++)
                    {
                        int pixelIndex = dropPosition + radiusOffset;
                        if (pixelIndex >= 0 && pixelIndex < length)
                        {
                            float normalizedRadius = radiusOffset / (float)(splashRadius + 1);
                            float hannBellWeight = 0.5f * (1.0f + MathF.Cos(normalizedRadius * MathF.PI));
                            this.heights[pixelIndex] += splashForce * hannBellWeight;
                            this.velocities[pixelIndex] += splashForce * SplashVelocityFactor * hannBellWeight;
                        }
                    }
                }
            }

            float fade = Math.Clamp(dynamicSettings.Fade, MinFade, MaxFade);
            float velocityDamping = MathF.Pow(1.0f - fade, 1.5f) * VelocityDampingScale + VelocityDampingBase;
            float heightDecay = MathF.Pow(1.0f - fade, 2.0f) * HeightDecayScale;
            float restoringForce = RestoringForceScale * MathF.Max(MinRestoringFade, fade);

            int subSteps = Math.Clamp((int)(dynamicSettings.Speed / SpeedSubStepDivisor) + 1, MinSubSteps, MaxSubSteps);
            bool isBouncy = dynamicSettings.BouncyWave;

            for (int stepIndex = 0; stepIndex < subSteps; stepIndex++)
            {
                this.laplacians[0] = isBouncy ? 2f * (this.heights[1] - this.heights[0]) : (this.heights[1] - this.heights[0]);
                this.laplacians[length - 1] = isBouncy ? 2f * (this.heights[length - 2] - this.heights[length - 1]) : (this.heights[length - 2] - this.heights[length - 1]);

                for (int i = 1; i < length - 1; i++)
                {
                    this.laplacians[i] = this.heights[i - 1] - 2f * this.heights[i] + this.heights[i + 1];
                }

                for (int i = 0; i < length; i++)
                {
                    this.velocities[i] = (this.velocities[i] * (1.0f - velocityDamping)) + (WavePropagationSpeedSquared * this.laplacians[i]) - (restoringForce * this.heights[i]);
                    this.heights[i] = (this.heights[i] + this.velocities[i]) * (1.0f - heightDecay);
                }

                if (!isBouncy)
                {
                    this.heights[0] *= BoundaryDampingFactor;
                    this.velocities[0] *= BoundaryDampingFactor;
                    this.heights[length - 1] *= BoundaryDampingFactor;
                    this.velocities[length - 1] *= BoundaryDampingFactor;
                }
            }

            float maxActivity = 0.0f;
            for (int i = 0; i < length; i++)
            {
                float height = this.heights[i];
                float absoluteHeight = MathF.Abs(height);
                if (absoluteHeight > maxActivity)
                    maxActivity = absoluteHeight;

                float crest = MathF.Max(0.0f, height);
                float trough = MathF.Max(0.0f, -height);
                float slope = (i > 0 && i < length - 1)
                    ? MathF.Abs(this.heights[i + 1] - this.heights[i - 1]) * SlopeWeight
                    : 0.0f;

                float elevation = (crest * CrestWeight) + (trough * TroughWeight) + slope;
                float pixelValue = MathHelper.Clamp(elevation, 0.0f, 1.0f);

                float cutoff = Math.Clamp(MinCutoff + (1.0f - fade) * CutoffFadeMultiplier, MinCutoff, MaxCutoff);
                if (pixelValue < cutoff)
                    pixelValue = 0.0f;

                prevStrip.LedPixels[i].Value = pixelValue;
            }

            int settleTimeoutMs = (int)(fade * SettleTimeoutFadeMultiplier + SettleTimeoutBaseMs);
            if (maxActivity < SettleMaxActivityThreshold && (currentTicksMs - this.lastSplashTime > settleTimeoutMs))
            {
                Array.Clear(this.heights, 0, length);
                Array.Clear(this.velocities, 0, length);
            }

            return prevStrip;
        }

        public void EnableEffect()
        {
            this.Reset();
        }

        public void DisableEffect()
        {
            this.Reset();
        }

        public void Reset()
        {
            this.heights = [];
            this.velocities = [];
            this.laplacians = [];
            this.lastSplashTime = 0;
            this.lastAudioValue = 0f;
        }
    }
}
