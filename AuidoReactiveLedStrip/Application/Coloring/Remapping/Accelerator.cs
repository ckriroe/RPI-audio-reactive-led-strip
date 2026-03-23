using Application.Settings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public class Accelerator : IRemapper
    {
        private Color[] remapped = [];

        public Color[] Remap(Color[] colors, DynamicSettings dynamicSettings)
        {
            if (dynamicSettings.Acceleration == 1.0f)
                return colors;

            if (this.remapped.Length != colors.Length)
            {
                this.remapped = new Color[colors.Length];
            }

            Color[] localRemapped = this.remapped;
            int center = dynamicSettings.EffectOrigin;
            int length = colors.Length;

            for (int i = 0; i < length; i++)
            {
                int d = i - center;

                int half = d < 0 ? center : (length - 1 - center);

                if (half == 0)
                {
                    localRemapped[i] = colors[center];
                    continue;
                }

                float normalized = d / (float)half;
                float warped = MathF.Sign(normalized) *
                               MathF.Pow(MathF.Abs(normalized), dynamicSettings.Acceleration);

                int mapped = center + (int)MathF.Round(warped * half);
                localRemapped[i] = colors[Math.Clamp(mapped, 0, length - 1)];
            }

            return localRemapped;
        }
    }
}
