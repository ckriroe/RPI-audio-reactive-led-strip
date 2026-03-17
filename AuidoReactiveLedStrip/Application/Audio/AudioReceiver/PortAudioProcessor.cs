using Application.Audio.AudioReceiver;
using AudioProcessing.AudioStream;
using PortAudioSharp;

namespace AudioProcessing.AudioProcessor
{
    public class PortAudioProcessor : IAudioReceiver
    {
        private Action<float[]>? audioCallback = null;

        private int? audioDeviceId = null;
        private int? channelCount = null;
        private int? bufferSize = null;
        private int? sampleRate = null;

        private bool isRunnging = false;
        private volatile bool isAudioThreadRunning = false;
        private Thread? audioWorker = null;
        private PortAudioBlockingAudioStream? currentAudioStream = null;

        public void Initialize(int audioDeviceId, int channelCount, int bufferSize, int sampleRate)
        {
            bool wasRunning = this.isRunnging;
            Action<float[]>? prevAudioCallback = this.audioCallback;

            if (wasRunning)
                this.StopAudioStream();

            this.audioDeviceId = audioDeviceId;
            this.channelCount = channelCount;
            this.bufferSize = bufferSize;
            this.sampleRate = sampleRate;

            if (wasRunning && prevAudioCallback != null)
                this.StartAudioStream(prevAudioCallback);
        }

        public void StartAudioStream(Action<float[]> audioCallback)
        {
            if (this.audioDeviceId == null || this.channelCount == null || this.bufferSize == null || this.sampleRate == null)
                throw new InvalidOperationException($"{nameof(PortAudioProcessor)} was not initialized before it was started");

            if (this.isRunnging)
                this.StopAudioStream();

            this.isRunnging = true;
            PortAudio.Initialize();
            DeviceInfo info = PortAudio.GetDeviceInfo(this.audioDeviceId.Value);

            var param = new StreamParameters();
            param.device = this.audioDeviceId.Value;
            param.channelCount = this.channelCount.Value;
            param.sampleFormat = SampleFormat.Float32;
            param.suggestedLatency = info.defaultLowInputLatency;
            param.hostApiSpecificStreamInfo = IntPtr.Zero;

            this.currentAudioStream = new PortAudioBlockingAudioStream(
                inParams: param,
                outParams: null,
                sampleRate: this.sampleRate.Value,
                framesPerBuffer: (uint) this.bufferSize.Value,
                streamFlags: StreamFlags.ClipOff,
                callback: null,
                userData: IntPtr.Zero
            );

            this.currentAudioStream.Start();
            this.audioCallback = audioCallback;
            this.audioWorker = new Thread(AudioProcessingLoop);
            this.isAudioThreadRunning = true;
            this.audioWorker.Start();            
        }

        public void StopAudioStream()
        {
            if (!this.isRunnging)
                return;

            this.isAudioThreadRunning = false;
            this.audioWorker?.Join();
            this.audioWorker = null;
            this.audioCallback = null;
            this.currentAudioStream?.Stop();
            this.currentAudioStream = null;

            PortAudio.Terminate();
            this.isRunnging = false;
        }

        public void AudioProcessingLoop()
        {
            while (this.isAudioThreadRunning)
            {
                int? bufferSize = this.bufferSize;
                if (bufferSize == null)
                    continue;

                float[]? readSamples = this.currentAudioStream?.ReadSamples(bufferSize.Value);
                if (readSamples != null)
                {
                    this.audioCallback?.Invoke(readSamples);
                }
            }
        }
    }
}
