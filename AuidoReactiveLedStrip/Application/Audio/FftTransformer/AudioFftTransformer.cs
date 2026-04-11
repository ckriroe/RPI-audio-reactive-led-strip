using MathNet.Numerics.IntegralTransforms;
using System.Numerics;

namespace Application.Audio.FftTransformer
{
    public class AudioFftTransformer : IAudioFftTransformer
    {
        public float[] ProcessAudioSamples(float[] samples, int channels)
        {
            int frames = samples.Length / channels;

            Complex[] fftInput = new Complex[frames];
            for (int f = 0; f < frames; f++)
            {
                float maxSample = float.MinValue;
                for (int c = 0; c < channels; c++)
                {
                    float sample = samples[f * channels + c];
                    if (Math.Abs(sample) > maxSample)
                        maxSample = Math.Abs(sample);
                }

                fftInput[f] = new Complex(maxSample, 0.0);
            }

            Fourier.Forward(fftInput, FourierOptions.Matlab);

            int magSize = frames - 1;
            float[] mags = new float[magSize];
            for (int i = 0; i < magSize; i++)
            {
                mags[i] = ((float)fftInput[i + 1].Magnitude / frames) * 512;
            }

            return mags;
        }
    }
}
