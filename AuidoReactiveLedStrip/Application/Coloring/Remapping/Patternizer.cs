using Application.Domain;
using Application.Settings;
using System.Drawing;

namespace Application.Coloring.Remapping
{
    public class Patternizer : IRemapper
    {
        public Color[] Remap(Color[] input, DynamicSettings dynamicSettings)
        {
            if (input == null || input.Length == 0)
            {
                return [];
            }

            int splits = dynamicSettings.PatternSplits;
            int flip = dynamicSettings.PatternFlip;
            int center = dynamicSettings.PatternCenter;
            int spread = dynamicSettings.PatternSpread;
            float sizeMod = dynamicSettings.PatternSectionSizeMod;

            if (splits == 0 && dynamicSettings.PatternFlip == -1)
            {
                return input;
            }

            int n = input.Length;
            int actualSplits = splits <= 1 ? 1 : Math.Min(splits, n);

            int centerPixel = center < 0 ? n / 2 : Math.Clamp(center, 0, n - 1);
            int c = (int)((long)centerPixel * actualSplits / n);
            c = Math.Clamp(c, 0, actualSplits - 1);

            // 1. Calculate weights and determine which sections survive
            List<int> activeIndices = new List<int>();
            float[] weights = new float[actualSplits];

            for (int i = 0; i < actualSplits; i++)
            {
                int dist = Math.Abs(i - c);
                float w = 1.0f + (dist * sizeMod);

                // If a negative sizeMod shrinks a section to 0 or less, it is discarded
                if (w > 0)
                {
                    weights[i] = w;
                    activeIndices.Add(i);
                }
            }

            // Safety: If there are more active sections than actual LEDs, drop the outermost ones
            if (activeIndices.Count > n)
            {
                activeIndices = activeIndices
                    .OrderBy(i => Math.Abs(i - c)) // Prioritize chunks closest to the center
                    .Take(n)
                    .OrderBy(i => i) // Restore original left-to-right order
                    .ToList();
            }

            // 2. Distribute exact integers ensuring a minimum size of 1 for active chunks
            int[] sizes = new int[actualSplits];
            int k = activeIndices.Count;

            if (k > 0)
            {
                // Guarantee every active section at least 1 pixel
                foreach (int i in activeIndices) sizes[i] = 1;
                int remainingPixels = n - k;

                if (remainingPixels > 0)
                {
                    float totalWeight = activeIndices.Sum(i => weights[i]);
                    var remainders = new List<(int Index, int Dist, float Rem)>();
                    int allocated = 0;

                    // Allocate remaining pixels proportionally
                    foreach (int i in activeIndices)
                    {
                        float exactAdd = (weights[i] / totalWeight) * remainingPixels;
                        int intAdd = (int)Math.Floor(exactAdd);
                        sizes[i] += intAdd;
                        allocated += intAdd;

                        // Track the fractional remainder for perfectly symmetric tie-breaking
                        remainders.Add((i, Math.Abs(i - c), exactAdd - intAdd));
                    }

                    int shortfall = remainingPixels - allocated;

                    // Group by distance from center so symmetric pairs share the same priority
                    var groups = remainders.GroupBy(x => x.Dist)
                                           .OrderByDescending(g => g.First().Rem)
                                           .ToList();

                    foreach (var group in groups)
                    {
                        if (shortfall <= 0) break;
                        var items = group.ToList();

                        if (shortfall >= items.Count)
                        {
                            // Add to both symmetric chunks
                            foreach (var item in items) sizes[item.Index]++;
                            shortfall -= items.Count;
                        }
                        else
                        {
                            // Unavoidable asymmetry (usually only 1 pixel off) due to odd remainder
                            for (int i = 0; i < shortfall; i++) sizes[items[i].Index]++;
                            shortfall = 0;
                        }
                    }
                }
            }

            // 3. Extract the chunks
            List<Color[]> activeChunks = new List<Color[]>();
            int newCenterIndex = -1;
            int currentStart = 0;

            for (int i = 0; i < actualSplits; i++)
            {
                if (sizes[i] > 0)
                {
                    if (i == c) newCenterIndex = activeChunks.Count;

                    Color[] chunk = new Color[sizes[i]];
                    for (int j = 0; j < sizes[i]; j++)
                    {
                        chunk[j] = input[currentStart + j];
                    }

                    // Apply flip based on the original logical distance
                    int dist = Math.Abs(i - c);
                    if (flip >= 0 && (dist + 1) % (flip + 1) == 0)
                    {
                        Array.Reverse(chunk);
                    }

                    activeChunks.Add(chunk);
                    currentStart += sizes[i];
                }
            }

            // 4. Apply Spread (Adjusted for potentially dropped sections)
            int newSplits = activeChunks.Count;
            int[] order = new int[newSplits];
            for (int i = 0; i < newSplits; i++) order[i] = i;

            if (spread > 0 && newSplits > 1 && newCenterIndex >= 0)
            {
                int leftCount = newCenterIndex;
                int rightCount = newSplits - 1 - newCenterIndex;

                int[] GetSpreadPattern(int length, int sprd)
                {
                    if (length <= 0) return [];
                    List<int> available = [.. Enumerable.Range(1, length)];
                    int[] pattern = new int[length];
                    int currentIndex = 0;

                    for (int i = 0; i < length; i++)
                    {
                        currentIndex = (currentIndex + sprd) % available.Count;
                        pattern[i] = available[currentIndex];
                        available.RemoveAt(currentIndex);
                    }
                    return pattern;
                }

                if (leftCount > 0)
                {
                    int[] leftPattern = GetSpreadPattern(leftCount, spread);
                    for (int i = 0; i < leftCount; i++)
                    {
                        order[newCenterIndex - 1 - i] = newCenterIndex - leftPattern[i];
                    }
                }

                if (rightCount > 0)
                {
                    int[] rightPattern = GetSpreadPattern(rightCount, spread);
                    for (int i = 0; i < rightCount; i++)
                    {
                        order[newCenterIndex + 1 + i] = newCenterIndex + rightPattern[i];
                    }
                }
            }

            // 5. Reassemble
            Color[] result = new Color[n];
            int pos = 0;
            for (int i = 0; i < newSplits; i++)
            {
                Color[] chunk = activeChunks[order[i]];
                chunk.CopyTo(result, pos);
                pos += chunk.Length;
            }

            return result;
        }
    }
}
