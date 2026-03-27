using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public class Repeater : IRemapper
    {
        public Color[] Remap(Color[] colors, DynamicSettings dynamicSettings)
        {
            if (dynamicSettings.EffectRepeats == 1)
                return colors;

            return this.RepeatArray(colors, dynamicSettings.EffectRepeats, dynamicSettings.PhysicalLedCount);
        }

        public Color[] RepeatArray(Color[] source, int times, int minLength)
        {
            int len = source.Length;
            Color[] result = new Color[minLength];

            for (int i = 0; i < times; i++)
            {
                Array.Copy(source, 0, result, i * len, len);
            }

            for (int i = source.Length * times; i < minLength; i++)
            {
                result[i] = Color.Black;
            }

            return result;
        }
    }
}
