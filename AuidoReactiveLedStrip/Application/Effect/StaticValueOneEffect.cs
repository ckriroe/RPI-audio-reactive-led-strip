namespace Application.Effect
{
    public class StaticValueOneEffect : StaticEffect
    {
        protected override float GetValueForIndex(int index, int length)
        {
            return 1.0f;
        }
    }
}
