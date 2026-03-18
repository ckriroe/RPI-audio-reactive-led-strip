using MathNet.Numerics.IntegralTransforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.FftTransformer
{
    public class AudioFftTransformer : IAudioFftTransformer
    {
        private int? channelCount;

        public void Initialize(int channelCount)
        {
            this.channelCount = channelCount;
        }

        public float[]? ProcessAudioSamples(float[] samples)
        {
            int? channels = this.channelCount;
            if (channels == null)
                return null;

            int frames = samples.Length / channels.Value;

            Complex[] fftInput = new Complex[frames];
            for (int f = 0; f < frames; f++)
            {
                float maxSample = float.MinValue;
                for (int c = 0; c < channels; c++)
                {
                    float sample = samples[f * channels.Value + c];
                    if (Math.Abs(sample) > maxSample) maxSample = Math.Abs(sample);
                }

                fftInput[f] = new Complex(maxSample, 0.0);
            }

            Fourier.Forward(fftInput, FourierOptions.Matlab);

            int halfSize = frames / 2 + 1;
            float[] mags = new float[halfSize];
            for (int i = 0; i < halfSize; i++)
            {
                mags[i] = (float)fftInput[i].Magnitude;
            }

            return mags;
        }
    }
}
