using Application.Settings;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application.Looper
{
    public interface ILooperConsumer
    {
        public void OnSettingsChanged(StaticSettings staticSettings, DynamicSettings dynamicSettings);
        public void OnTick();
    }
}
