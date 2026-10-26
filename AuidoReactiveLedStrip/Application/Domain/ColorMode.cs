namespace Application.Domain
{
    public enum ColorMode
    {
        ColorByEffectValue = 0,
        ColorByIndex = 1,
        ColorByDistanceToCenter = 2,
        ColorByDistanceToBorder = 3,
        ColorWave = 4,
        ColorIslands = 5,
        ColorCycleOnBlank = 6,
        ColorCycleByTime = 7,
        ColorCycleByAudio = 8
    }
}
