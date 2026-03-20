namespace Application.Effect
{
    public interface IStatefulEffect : IEffect
    {
        void EnableEffect();

        void DisableEffect();
    }
}
