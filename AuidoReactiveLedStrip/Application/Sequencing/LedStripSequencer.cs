using Application.LedStripRendering;
using Application.RuntimeSettings;
using Application.Util;
using System.Drawing;

namespace Application.Sequencing
{
    public enum LedSequenceState
    {
        None,
        UseCurrentEffect,
        WarumupNextEffect,
        Transition,
        NextEffectSwitch
    }

    public class LedStripSequencer : ILedStripSequencer
    {
        private ILedStripRenderer mainRenderer;
        private ILedStripRenderer transitionRenderer;

        private StaticSettings? staticSettings = null;
        private DynamicPresetSettings? dynamicPresetSettings = null;

        private Preset? currentPreset = null;
        private Preset? nextPreset = null;
        private long referenceTimestamp = 0;
        private bool isTransitioning = false;
        private LedSequenceState currentState = LedSequenceState.None;

        public LedStripSequencer(ILedStripRenderer mainRenderer, ILedStripRenderer transitionRenderer)
        {
            this.mainRenderer = mainRenderer;
            this.transitionRenderer = transitionRenderer;
        }

        public Color[]? RenderLedStrip()
        {
            StaticSettings? staticSettings = this.staticSettings;
            DynamicPresetSettings? dynamicPresetSettings = this.dynamicPresetSettings;
            if (staticSettings == null || dynamicPresetSettings == null)
                return null;

            long timestamp = Environment.TickCount;
            if (this.currentPreset == null)
                this.RestartSequence(dynamicPresetSettings, timestamp);

            Preset? currentSetting = this.currentPreset;
            Preset? nextSettings = this.nextPreset;
            if (currentSetting == null)
                return null;

            long deltaTime = timestamp - this.referenceTimestamp;
            if (deltaTime < currentSetting.EffectSettings.EffectDurationMs || nextSettings == null)
            {
                return this.GetCurrentEffect(staticSettings, currentSetting, nextSettings, deltaTime);
            }
            else if (deltaTime < currentSetting.EffectSettings.EffectDurationMs + currentSetting.EffectSettings.EffectTransitionDurationMs)
            {
                return this.GetTransitionEffect(staticSettings, timestamp, currentSetting, nextSettings);
            }
            else
            {
                return this.CompleteTransition(staticSettings, dynamicPresetSettings, timestamp, nextSettings);
            }
        }

        public void ApplySettings(StaticSettings staticSettings, DynamicPresetSettings dynamicPresetSettings)
        {
            this.staticSettings = staticSettings;
            this.dynamicPresetSettings = dynamicPresetSettings;

            Preset? currentPreset = this.currentPreset;
            if (currentPreset != null)
            {
                this.SetPresets(dynamicPresetSettings, currentPreset.Id);
                this.mainRenderer.ApplySettings(staticSettings, currentPreset.EffectSettings);

                Preset? nextPreset = this.nextPreset;
                if (nextPreset != null && this.isTransitioning)
                    this.transitionRenderer.ApplySettings(staticSettings, nextPreset.EffectSettings);
            }
        }

        public void Disable()
        {
            this.mainRenderer.Disable();
            this.transitionRenderer.Disable();
        }

        private Color[]? GetCurrentEffect(StaticSettings staticSettings, Preset currentSetting, Preset? nextSettings, long deltaTime)
        {
            if (nextSettings != null && deltaTime > currentSetting.EffectSettings.EffectDurationMs - currentSetting.EffectSettings.EffectTransitionWarmupDuration)
            {
                if (!this.isTransitioning)
                {
                    this.isTransitioning = true;
                    this.transitionRenderer.ApplySettings(staticSettings, nextSettings.EffectSettings);
                }

                if (this.currentState != LedSequenceState.WarumupNextEffect)
                {
                    this.currentState = LedSequenceState.WarumupNextEffect;
                    if (staticSettings.PrintSequenceInfos)
                        Console.WriteLine($"Sequencer: Warmup next effect. Current effect: '{currentSetting.Name}' ({currentSetting.Id}), next effect: '{nextSettings.Name}' ({nextSettings.Id})");
                }                    

                this.transitionRenderer.RenderLedStrip();
            }

            if (this.currentState != LedSequenceState.UseCurrentEffect && this.currentState != LedSequenceState.WarumupNextEffect)
            {
                this.currentState = LedSequenceState.UseCurrentEffect;
                if (staticSettings.PrintSequenceInfos)
                    Console.WriteLine($"Sequencer: Calculate effect for '{currentSetting.Name}' ({currentSetting.Id})");
            }

            return this.mainRenderer.RenderLedStrip();
        }

        private Color[]? GetTransitionEffect(StaticSettings staticSettings, long timestamp, Preset currentSetting, Preset nextSettings)
        {
            if (!this.isTransitioning)
            {
                this.isTransitioning = true;
                this.transitionRenderer.ApplySettings(staticSettings, nextSettings.EffectSettings);
            }

            if (this.currentState != LedSequenceState.Transition)
            {
                this.currentState = LedSequenceState.Transition;
                if (staticSettings.PrintSequenceInfos)
                    Console.WriteLine($"Sequencer: transitioning to next effect. Current effect: '{currentSetting.Name}' ({currentSetting.Id}), next effect: '{nextSettings.Name}' ({nextSettings.Id})");
            }

            long start = this.referenceTimestamp + currentSetting.EffectSettings.EffectDurationMs;
            long end = start + currentSetting.EffectSettings.EffectTransitionDurationMs;

            float t = currentSetting.EffectSettings.EffectTransitionDurationMs == 0
                ? 1f
                : (timestamp - start) / (float)currentSetting.EffectSettings.EffectTransitionDurationMs;

            t = Math.Clamp(t, 0f, 1f);

            Color[]? mainColors = this.mainRenderer.RenderLedStrip();
            Color[]? transitionColors = this.transitionRenderer.RenderLedStrip();

            return ColorHelper.LerpColors(mainColors, transitionColors, t);
        }

        private Color[]? CompleteTransition(StaticSettings staticSettings, DynamicPresetSettings dynamicPresetSettings, long timestamp, Preset nextSettings)
        {
            if (!this.isTransitioning)
            {
                this.transitionRenderer.ApplySettings(staticSettings, nextSettings.EffectSettings);
            }

            if (this.currentState != LedSequenceState.NextEffectSwitch)
            {
                this.currentState = LedSequenceState.NextEffectSwitch;
                if (staticSettings.PrintSequenceInfos)
                    Console.WriteLine($"Sequencer: Switching to next effect: '{nextSettings.Name}' ({nextSettings.Id})");
            }

            ILedStripRenderer tmp = this.mainRenderer;
            this.mainRenderer = this.transitionRenderer;
            this.transitionRenderer = tmp;
            this.transitionRenderer.Disable();

            this.SetPresets(dynamicPresetSettings, this.nextPreset?.Id);
            this.isTransitioning = false;
            this.referenceTimestamp = timestamp;
            return this.mainRenderer.RenderLedStrip();
        }

        private void RestartSequence(DynamicPresetSettings dynamicPresetSettings, long timestamp)
        {
            this.SetPresets(dynamicPresetSettings, null);
            this.referenceTimestamp = timestamp;
            this.isTransitioning = false;
        }

        private void SetPresets(DynamicPresetSettings dynamicPresetSettings, Guid? currentPresetId)
        {
            this.currentPreset = dynamicPresetSettings.Presets.FirstOrDefault(p => currentPresetId != null && p.Id == currentPresetId) ??
                dynamicPresetSettings.Presets.FirstOrDefault(p => p.Id == dynamicPresetSettings.SelectedPresetId) ??
                dynamicPresetSettings.Presets.FirstOrDefault();

            this.nextPreset = this.currentPreset?.EffectSettings.NextEffectId != null ?
                dynamicPresetSettings.Presets.FirstOrDefault(p => p.Id == this.currentPreset.EffectSettings.NextEffectId.Value) :
                null;
        }
    }
}
