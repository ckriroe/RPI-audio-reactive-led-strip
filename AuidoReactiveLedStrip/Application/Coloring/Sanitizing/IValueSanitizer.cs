using Application.Settings;

namespace Application.Coloring.Sanitizing
{
    public interface IValueSanitizer
    {
        float SanitizeValue(float value, StaticSettings staticSettings);
    }
}
