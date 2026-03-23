using Application.Settings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public interface IRemapper
    {
        Color[] Remap(Color[] colors, DynamicSettings dynamicSettings);
    }
}
