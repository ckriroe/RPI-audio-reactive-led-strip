using Application.RuntimeSettings;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Looper
{
    public interface ILooperConsumer
    {
        public void OnSettingsChanged(StaticSettings staticSettings, DynamicPresetSettings dynamicSettings);
        public void OnTick();
        public void OnBeforeTick();
    }
}
