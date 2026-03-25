using Application.RuntimeSettings;
using Application.Util;

namespace Application.Coloring.Noise
{
    public class NoiseGenerator : INoiseGenerator
    {
        private float[] ledNoise = [];

        public float GetSmoothNoise(int index, int length, DynamicSettings dynamicSettings)
        {
            if (this.ledNoise.Length != length)
                this.ledNoise = new float[length];

            var noiseAmount = dynamicSettings.NoiseAmount;
            if (noiseAmount == 0.0f)
                return 0.0f;

            var noiseSmoothing = dynamicSettings.NoiseSmoothing;
            float target = MathHelper.NextFloatSigned() * noiseAmount;

            var newNoise = ledNoise[index] * noiseSmoothing +
                target * (1.0f - noiseSmoothing);

            ledNoise[index] = newNoise;
            return newNoise;
        }
    }
}
