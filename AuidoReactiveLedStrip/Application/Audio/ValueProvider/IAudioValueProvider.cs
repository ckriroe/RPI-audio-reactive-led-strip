using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.ValueProvider
{
    public interface IAudioValueProvider : IAudioDataProvider
    {

        float GetAudioValue();
    }
}
