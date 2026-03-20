using Application.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Audio.ValueProvider
{
    public class SimpleAudioValueProvider : BaseAudioValueProvider
    {
        protected override void CalculateAudioValue(float maxFrequency)
        {
            if (base.settings == null)
                return;

            float frequencyDiff = base.settings.MaxFrequencyAmplitude - base.settings.MinFrequencyAmplitude;
            if (frequencyDiff <= 0.0f)
            {
                base.currentValue = 0.0f;
                return;
            }

            base.currentValue = MathHelper.Clamp((maxFrequency - base.settings.MinFrequencyAmplitude) / frequencyDiff, 0.0f, 1.0f);
        }
    }
}
