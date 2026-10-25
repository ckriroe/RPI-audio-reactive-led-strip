using Application.Util;
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

                double hannWindow = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * f / (frames - 1)));
                fftInput[f] = new Complex(maxSample * hannWindow, 0.0);
            }

            Fourier.Forward(fftInput, FourierOptions.Matlab);
            int magSize = frames - 1;
            float[] mags = new float[magSize];
            float scalingFactor = 4.0f / frames;

            for (int i = 0; i < magSize; i++)
            {
                mags[i] = (float)fftInput[i + 1].Magnitude * scalingFactor;
            }

            return mags;
        }
    }
}
