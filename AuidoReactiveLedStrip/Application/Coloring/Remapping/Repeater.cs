using Application.Settings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public class Repeater : IRemapper
    {
        public Color[] Remap(Color[] colors, DynamicSettings dynamicSettings)
        {
            if (dynamicSettings.EffectRepeats == 1)
                return colors;

            return this.RepeatArray(colors, dynamicSettings.EffectRepeats);
        }

        public Color[] RepeatArray(Color[] source, int times)
        {
            int len = source.Length;
            Color[] result = new Color[len * times];

            for (int i = 0; i < times; i++)
            {
                Array.Copy(source, 0, result, i * len, len);
            }

            return result;
        }
    }
}
