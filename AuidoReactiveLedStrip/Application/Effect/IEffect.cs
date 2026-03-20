using Application.Domain;

namespace Application.Effect
{
    public interface IEffect
    {
        bool IsStatic { get; }

        bool UseAudioFft { get; }

        bool UseAudioValue { get; }

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length);
    }
}
