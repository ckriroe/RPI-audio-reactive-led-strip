using Application.Audio.Receiver;
using Application.RuntimeSettings;
using Application.Util;
using AudioProcessing.Stream;
using PortAudioSharp;
using System.Buffers;
using System.Threading.Channels;

namespace AudioProcessing.AudioProcessor
{
    public class AudioRegistration
    {
        public required Guid Identifier { get; init; }
        public required int RequestedBufferSize { get; init; }
        public required Action<float[], int> AudioCallback { get; init; }
        public RingBuffer<float> Buffer { get; init; } = new RingBuffer<float>(1024);
    }

    public class PortAudioReceiver : IAudioReceiver, IDisposable
    {
        private record AudioState(
            AudioRegistration[] Registrations,
            int? MinBufferSize,
            int? ChannelCount
        );

        private readonly Channel<Action> commandChannel = Channel.CreateUnbounded<Action>();
        private readonly CancellationTokenSource controlCts = new CancellationTokenSource();
        private readonly Task controlTask;

        private volatile AudioState currentAudioState = new AudioState(Array.Empty<AudioRegistration>(), null, null);
        private readonly Dictionary<Guid, AudioRegistration> activeRegistrations = new Dictionary<Guid, AudioRegistration>();

        private bool isRunning = false;
        private int? audioDeviceId = null;
        private int? channelCount = null;
        private int? sampleRate = null;
        private int? minBufferSize = null;

        private volatile bool isAudioThreadRunning = false;
        private Thread? audioWorker = null;
        private PortAudioBlockingAudioStream? currentAudioStream = null;

        public PortAudioReceiver()
        {
            this.controlTask = Task.Run(ControlLoopAsync);
        }

        public void ApplyStaticSettings(StaticSettings staticSettings)
        {
            this.commandChannel.Writer.TryWrite(() =>
            {
                if (this.audioDeviceId == staticSettings.AudioDeviceId &&
                    this.channelCount == staticSettings.Channels &&
                    this.sampleRate == staticSettings.SampleRate)
                    return;

                bool wasRunning = this.isRunning;
                if (wasRunning)
                    this.StopAudioStream();

                this.audioDeviceId = staticSettings.AudioDeviceId;
                this.channelCount = staticSettings.Channels;
                this.sampleRate = staticSettings.SampleRate;

                this.PublishStateSnapshot();

                if (wasRunning || this.ShouldAudioStreamBeRunning())
                    this.StartAudioStream();
            });
        }

        public void RegisterAudioConsumer(Guid identifier, int requestedBufferSize, Action<float[], int> audioCallback)
        {
            this.commandChannel.Writer.TryWrite(() =>
            {
                if (this.activeRegistrations.TryGetValue(identifier, out var existing))
                {
                    this.activeRegistrations[identifier] = new AudioRegistration
                    {
                        Identifier = identifier,
                        AudioCallback = audioCallback,
                        RequestedBufferSize = requestedBufferSize,
                        Buffer = existing.Buffer
                    };
                }
                else
                {
                    this.activeRegistrations.Add(identifier, new AudioRegistration
                    {
                        Identifier = identifier,
                        AudioCallback = audioCallback,
                        RequestedBufferSize = requestedBufferSize
                    });
                }

                this.ReCalcMinBufferSize();
                this.PublishStateSnapshot();

                if (!this.isRunning && this.ShouldAudioStreamBeRunning())
                    this.StartAudioStream();
            });
        }

        public void UnregisterAudioConsumer(Guid identifier)
        {
            this.commandChannel.Writer.TryWrite(() =>
            {
                if (this.activeRegistrations.Remove(identifier))
                {
                    this.ReCalcMinBufferSize();
                    this.PublishStateSnapshot();

                    if (this.isRunning && !this.ShouldAudioStreamBeRunning())
                        this.StopAudioStream();
                }
            });
        }

        public void Dispose()
        {
            this.controlCts.Cancel();
            this.commandChannel.Writer.Complete();

            if (this.controlTask != null)
            {
                try
                {
                    this.controlTask.GetAwaiter().GetResult();
                }
                catch
                {
                    // Ignore
                }
            }

            if (this.isRunning)
            {
                this.StopAudioStream();
            }
        }


        private async Task ControlLoopAsync()
        {
            try
            {
                await foreach (var command in this.commandChannel.Reader.ReadAllAsync(this.controlCts.Token))
                {
                    command(); // Execute the queued modification sequentially
                }
            }
            catch (OperationCanceledException) { /* Clean teardown */ }
        }

        private void PublishStateSnapshot()
        {
            // Atomically swaps the reference. The audio loop reads this entirely lock-free.
            this.currentAudioState = new AudioState(
                this.activeRegistrations.Values.ToArray(),
                this.minBufferSize,
                this.channelCount
            );
        }

        private void ReCalcMinBufferSize()
        {
            this.minBufferSize = this.activeRegistrations.Any() ?
                this.activeRegistrations.Values.Select(reg => reg.RequestedBufferSize).Min() :
                null;
        }

        private bool ShouldAudioStreamBeRunning() => this.activeRegistrations.Any();

        private void StartAudioStream()
        {
            if (this.audioDeviceId == null || this.channelCount == null ||
                this.minBufferSize == null || this.sampleRate == null || this.isRunning)
                return;

            this.isRunning = true;
            PortAudio.Initialize();
            DeviceInfo info = PortAudio.GetDeviceInfo(this.audioDeviceId.Value);

            var param = new StreamParameters
            {
                device = this.audioDeviceId.Value,
                channelCount = this.channelCount.Value,
                sampleFormat = SampleFormat.Float32,
                suggestedLatency = info.defaultLowInputLatency,
                hostApiSpecificStreamInfo = IntPtr.Zero
            };

            this.currentAudioStream = new PortAudioBlockingAudioStream(
                inParams: param,
                outParams: null,
                sampleRate: this.sampleRate.Value,
                framesPerBuffer: (uint)this.minBufferSize.Value,
                streamFlags: StreamFlags.ClipOff,
                callback: null,
                userData: IntPtr.Zero
            );

            this.currentAudioStream.Start();

            this.isAudioThreadRunning = true;
            this.audioWorker = new Thread(AudioProcessingLoop);
            this.audioWorker.Start();
            Console.WriteLine("Started audio processing");
        }

        private void StopAudioStream()
        {
            if (!this.isRunning)
                return;

            this.isAudioThreadRunning = false;
            this.audioWorker?.Join();
            this.audioWorker = null;

            this.currentAudioStream?.Stop();
            this.currentAudioStream?.Dispose();
            this.currentAudioStream = null;

            PortAudio.Terminate();
            Console.WriteLine("Disabled audio processing");
            this.isRunning = false;
        }

        private void AudioProcessingLoop()
        {
            while (this.isAudioThreadRunning)
            {
                AudioState state = this.currentAudioState;

                if (state.MinBufferSize == null || state.ChannelCount == null || state.Registrations.Length == 0)
                {
                    Thread.Sleep(5);
                    continue;
                }

                int bufferSize = state.MinBufferSize.Value;
                int channels = state.ChannelCount.Value;

                var stream = this.currentAudioStream;
                if (stream == null)
                    continue;

                float[]? readSamples = stream.ReadSamples(bufferSize);
                if (readSamples == null)
                    continue;

                int inputSamples = bufferSize * channels;

                foreach (var registration in state.Registrations)
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
