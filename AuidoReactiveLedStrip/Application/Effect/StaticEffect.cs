using Application.Domain;
using Application.Util;

namespace Application.Effect
{
    public abstract class StaticEffect : IEffect
    {
        public bool IsStatic => true;

        public bool UseAudioFft => false;

        public bool UseAudioValue => false;

        public LedStrip? RenderEffekt(LedStrip? prevStrip, int length)
        {
            prevStrip ??= LedHelper.CreateEmptyStrip(length);

            for (int i = 0; i < length; i++)
            {
                prevStrip.LedPixels[i].Value = this.GetValueForIndex(i, length);
            }

            return prevStrip;
        }

        protected abstract float GetValueForIndex(int index, int length);
    }
}
