using Application.RuntimeSettings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public class Repeater : IRemapper
    {
        public Color[] Remap(Color[] colors, DynamicEffectSettings dynamicSettings, StaticSettings staticSettings)
        {
            if (dynamicSettings.EffectRepeats == 1)
                return colors;

            return this.RepeatArray(colors, dynamicSettings.EffectRepeats, staticSettings.LedCount);
        }

        public Color[] RepeatArray(Color[] source, int times, int minLength)
        {
            int len = source.Length;
            Color[] result = new Color[minLength];

            for (int i = 0; i < times; i++)
            {
                if (i * len + (len - 1) >= result.Length)
                    break;

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
