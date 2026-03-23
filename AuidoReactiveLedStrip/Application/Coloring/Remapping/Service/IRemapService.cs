using Application.Domain;
using System.Drawing;

namespace Application.Coloring.Remapping.Service
{
    public interface IRemapService
    {
        Color[] RemapColors(LedStrip ledStrip);
    }
}