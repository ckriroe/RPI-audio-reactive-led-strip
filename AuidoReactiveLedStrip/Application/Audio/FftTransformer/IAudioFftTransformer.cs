namespace Application.Audio.FftTransformer
{
    public interface IAudioFftTransformer
    {
        float[] ProcessAudioSamples(float[] samples, int channels);
    }
}
