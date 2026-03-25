using Application.RuntimeSettings;

namespace Application.Coloring.Sanitizing
{
    public interface IValueSanitizer
    {
        float SanitizeValue(float value, StaticSettings staticSettings);
    }
}
