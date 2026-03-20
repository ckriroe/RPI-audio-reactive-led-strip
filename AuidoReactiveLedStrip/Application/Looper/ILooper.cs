namespace Application.Looper
{
    public interface ILooper
    {
        public void SetConsumer(ILooperConsumer consumer);
        public void StartLooper();
        public void StopLooper();
    }
}
