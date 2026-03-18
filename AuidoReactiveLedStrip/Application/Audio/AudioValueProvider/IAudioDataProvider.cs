using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.AudioValueProvider
{
    public interface IAudioDataProvider
    {
        void Initialize(BaseAudioDataProviderSettings settings);

        void SetNewFftData(float[] fftData);

        float[]? GetCurrentFftData();
    }
}
