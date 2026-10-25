using System;

namespace Application.Util
{
    public static class AudioUtil
    {
        public const float MinDb = -120.0f;
        public const float MaxDb = 0.0f;
        private const float MinMagnitude = 0.000001f; // 10^(-120/20)

        public static float MagnitudeToDb(float magnitude)
        {
            if (magnitude <= MinMagnitude)
                return MinDb;

            float db = 20.0f * MathF.Log10(magnitude);
            return Math.Clamp(db, MinDb, MaxDb);
        }

        public static float DbToMagnitude(float db)
        {
            if (db <= MinDb)
                return 0.0f;

            float clampedDb = Math.Clamp(db, MinDb, MaxDb);
            return MathF.Pow(10.0f, clampedDb / 20.0f);
        }
    }
}
