using Application.RuntimeSettings;

namespace Application.Coloring.Sanitizing
{
    public class ValueSanitizer : IValueSanitizer
    {
        public float SanitizeValue(float value, StaticSettings staticSettings)
        {
            if (value < staticSettings.MinSanitizedValue)
                return 0.0f;

            return value;
        }
    }
}
