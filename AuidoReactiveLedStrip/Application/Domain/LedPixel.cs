using System.Drawing;

namespace Application.Domain
{
    public class LedPixel
    {
        public LedPixel(float value) : this(value, Color.Black) {}

        public LedPixel(float value, Color color)
        {
            Value = value;
            Color = color;
        }

        public float Value { get; set; }

        public Color Color { get; set; }
    }
}
