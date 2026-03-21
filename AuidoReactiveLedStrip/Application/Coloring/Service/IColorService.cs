using Application.Domain;

namespace Application.Coloring.Service
{
    public interface IColorService
    {
        void ColorizeLedStrip(LedStrip ledStrip);
        void SetColorMode(ColorMode colorMode);
    }
}
