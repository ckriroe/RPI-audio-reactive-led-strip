namespace Application.Util
{
    public class LimitedBuffer<T>
    {
        private readonly int maxSize;
        private readonly Queue<T> queue;

        public LimitedBuffer(int maxSize)
        {
            this.maxSize = maxSize;
            this.queue = new Queue<T>();
        }

        public void Add(T item)
        {
            if (this.queue.Count == this.maxSize)
                this.queue.Dequeue();

            this.queue.Enqueue(item);
        }

        public IEnumerable<T> Items => this.queue;

        public int MaxSize => this.maxSize;
    }
}
