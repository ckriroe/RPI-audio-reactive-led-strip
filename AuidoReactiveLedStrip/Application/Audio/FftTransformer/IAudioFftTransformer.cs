namespace Application.Audio.FftTransformer
{
    public interface IAudioFftTransformer
    {
        void Initialize(int channelCount);

        float[]? ProcessAudioSamples(float[] samples);
    }
}
