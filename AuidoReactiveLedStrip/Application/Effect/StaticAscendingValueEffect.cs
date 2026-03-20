namespace Application.Effect
{
    public class StaticAscendingValueEffect : StaticEffect
    {
        protected override float GetValueForIndex(int index, int length)
        {
            return index / (float)(length - 1);
        }
    }
}
