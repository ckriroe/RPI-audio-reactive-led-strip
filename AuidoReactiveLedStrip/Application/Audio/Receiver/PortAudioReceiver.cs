using Application.Audio.Receiver;
using Application.RuntimeSettings;
using Application.Util;
using AudioProcessing.Stream;
using PortAudioSharp;
using System.Buffers;

namespace AudioProcessing.AudioProcessor
{
    public class AudioRegistration
    {
        public required Guid Identifier { get; set; }

        public required int RequestedBufferSize { get; set; }

        public required Action<float[], int> AudioCallback { get; set; }

        public RingBuffer<float> Buffer { get; } = new RingBuffer<float>(1024);
    }

    public class PortAudioReceiver : IAudioReceiver
    {
        private readonly object lck = new object();

        private Dictionary<Guid, AudioRegistration> activeAudioRegistrations = new Dictionary<Guid, AudioRegistration>();
        private bool isRunnging = false;
        private int? audioDeviceId = null;
        private int? channelCount = null;
        private int? sampleRate = null;
        private int? minBufferSize = null;

        private volatile bool isAudioThreadRunning = false;
        private Thread? audioWorker = null;
        private PortAudioBlockingAudioStream? currentAudioStream = null;

        public void ApplyStaticSettings(StaticSettings staticSettings)
        {
            if (this.audioDeviceId == staticSettings.AudioDeviceId &&
                this.channelCount == staticSettings.Channels &&
                this.sampleRate == staticSettings.SampleRate)
                return;

            bool wasRunning = this.isRunnging;
            if (wasRunning)
                this.StopAudioStream();

            lock (this.lck)
            {
                this.audioDeviceId = staticSettings.AudioDeviceId;
                this.channelCount = staticSettings.Channels;
                this.sampleRate = staticSettings.SampleRate;

                if (wasRunning || this.ShouldAudioStreamBeRunning())
                    this.StartAudioStream();
            }
        }

        public void RegisterAudioConsumer(Guid identifier, int requestedBufferSize, Action<float[], int> audioCallback)
        {
            lock (this.lck)
            {
                if (this.activeAudioRegistrations.TryGetValue(identifier, out AudioRegistration? existingRegistration) && existingRegistration != null)
                {
                    existingRegistration.AudioCallback = audioCallback;
                    existingRegistration.RequestedBufferSize = requestedBufferSize;
                }
                else
                {
                    this.activeAudioRegistrations.Add(identifier, new AudioRegistration()
                    {
                        Identifier = identifier,
                        AudioCallback = audioCallback,
                        RequestedBufferSize = requestedBufferSize
                    });
                }

                this.ReCalcMinBufferSize();

                if (!this.isRunnging && this.ShouldAudioStreamBeRunning())
                    this.StartAudioStream();
            }
        }

        public void UnregisterAudioConsumer(Guid identifier)
        {
            bool shouldStopAudioStream = false;
            lock (this.lck)
            {
                this.activeAudioRegistrations.Remove(identifier);
                this.ReCalcMinBufferSize();
                if (this.isRunnging && !this.ShouldAudioStreamBeRunning())
                    shouldStopAudioStream = true;
            }

            if (shouldStopAudioStream)
                this.StopAudioStream();
        }

        private void ReCalcMinBufferSize()
        {
            this.minBufferSize = this.activeAudioRegistrations.Any() ?
                this.activeAudioRegistrations.Values.Select(reg => reg.RequestedBufferSize).Min() :
                null;
        }

        private bool ShouldAudioStreamBeRunning()
        {
            return this.activeAudioRegistrations.Any();
        }

        private void StartAudioStream()
        {
            if (this.audioDeviceId == null ||
                this.channelCount == null ||
                this.minBufferSize == null ||
                this.sampleRate == null ||
                this.isRunnging)
                return;

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
                framesPerBuffer: (uint) this.minBufferSize.Value,
                streamFlags: StreamFlags.ClipOff,
                callback: null,
                userData: IntPtr.Zero
            );

            this.currentAudioStream.Start();
            this.audioWorker = new Thread(AudioProcessingLoop);
            this.isAudioThreadRunning = true;
            this.audioWorker.Start();
            Console.WriteLine("Stated audio processing");
        }

        private void StopAudioStream()
        {
            if (!this.isRunnging)
                return;

            this.isAudioThreadRunning = false;
            this.audioWorker?.Join();
            this.audioWorker = null;
            this.currentAudioStream?.Stop();
            this.currentAudioStream?.Dispose();
            this.currentAudioStream = null;

            PortAudio.Terminate();
            Console.WriteLine("Disabled audio processing");
            this.isRunnging = false;
        }

        private void AudioProcessingLoop()
        {
            while (this.isAudioThreadRunning)
            {
                lock (this.lck)
                {
                    if (this.minBufferSize == null ||
                        this.channelCount == null ||
                        !this.activeAudioRegistrations.Any())
                        continue;

                    int bufferSize = this.minBufferSize.Value;
                    int channels = this.channelCount.Value;
                    float[]? readSamples = this.currentAudioStream?.ReadSamples(bufferSize);
                    if (readSamples == null)
                        continue;

                    int inputSamples = bufferSize * channels;

                    foreach (var registration in this.activeAudioRegistrations.Values)
                    {
                        this.ProcessRegistration(
                            registration,
                            readSamples,
                            inputSamples,
                            channels,
                            bufferSize
                        );
                    }
                }
            }
        }

        private void ProcessRegistration(
            AudioRegistration registration,
            float[] readSamples,
            int inputSamples,
            int channels,
            int minBufferSize)
        {
            int targetSamples = registration.RequestedBufferSize * channels;
            if (registration.RequestedBufferSize == minBufferSize)
            {
                registration.AudioCallback(readSamples, channels);
                return;
            }

            var ring = registration.Buffer;

            ring.EnsureCapacity(targetSamples * 2);
            ring.Write(readSamples, inputSamples);

            while (ring.Count >= targetSamples)
            {
                float[] output = ArrayPool<float>.Shared.Rent(targetSamples);

                try
                {
                    ring.Read(output, targetSamples);
                    registration.AudioCallback(output, channels);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(output);
                }
            }
        }
    }
}
