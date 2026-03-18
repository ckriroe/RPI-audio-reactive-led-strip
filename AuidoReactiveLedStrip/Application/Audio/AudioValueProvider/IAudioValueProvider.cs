using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.AudioValueProvider
{
    public interface IAudioValueProvider : IAudioDataProvider
    {

        float GetAudioValue();
    }
}
