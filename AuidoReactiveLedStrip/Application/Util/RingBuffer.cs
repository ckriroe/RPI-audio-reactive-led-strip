namespace Application.Util
{
    public sealed class RingBuffer<T>
    {
        private T[] buffer;
        private int writeIndex;
        private int readIndex;
        private int count;

        public int Count => count;
        public int Capacity => buffer.Length;

        public RingBuffer(int initialCapacity)
        {
            if (initialCapacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(initialCapacity));

            this.buffer = new T[initialCapacity];
        }

        public void EnsureCapacity(int capacity)
        {
            if (this.buffer.Length >= capacity)
                return;

            var newBuffer = new T[capacity];

            if (this.count > 0)
            {
                if (this.readIndex < this.writeIndex)
                {
                    Array.Copy(this.buffer, this.readIndex, newBuffer, 0, this.count);
                }
                else
                {
                    int firstPart = this.buffer.Length - this.readIndex;
                    Array.Copy(this.buffer, this.readIndex, newBuffer, 0, firstPart);
                    Array.Copy(this.buffer, 0, newBuffer, firstPart, this.writeIndex);
                }
            }

            this.buffer = newBuffer;
            this.readIndex = 0;
            this.writeIndex = this.count;
        }

        public void Write(T[] source, int length)
        {
            if (length > source.Length)
                throw new ArgumentOutOfRangeException(nameof(length));

            for (int i = 0; i < length; i++)
            {
                this.buffer[this.writeIndex] = source[i];
                this.writeIndex = (this.writeIndex + 1) % this.buffer.Length;

                if (this.count < this.buffer.Length)
                {
                    this.count++;
                }
                else
                {
                    // overwrite oldest
                    this.readIndex = (this.readIndex + 1) % this.buffer.Length;
                }
            }
        }

        public void Read(T[] destination, int length)
        {
            if (length > this.count)
                throw new InvalidOperationException("Not enough data in buffer");

            if (this.readIndex + length <= this.buffer.Length)
            {
                Array.Copy(this.buffer, this.readIndex, destination, 0, length);
            }
            else
            {
                int firstPart = this.buffer.Length - this.readIndex;
                Array.Copy(this.buffer, this.readIndex, destination, 0, firstPart);
                Array.Copy(this.buffer, 0, destination, firstPart, length - firstPart);
            }

            this.readIndex = (this.readIndex + length) % this.buffer.Length;
            this.count -= length;
        }
    }
}
