namespace Application.Looper
{
    public class LooperPerformanceMetrics
    {
        public double ActualFps { get; }
        public double FrameTimeMs { get; }
        public double ProcessTimeMs { get; }
        public double TheoreticalFps { get; }

        public LooperPerformanceMetrics(double actualFps, double frameTimeMs, double processTimeMs, double theoreticalFps)
        {
            this.ActualFps = actualFps;
            this.FrameTimeMs = frameTimeMs;
            this.ProcessTimeMs = processTimeMs;
            this.TheoreticalFps = theoreticalFps;
        }
    }
}
