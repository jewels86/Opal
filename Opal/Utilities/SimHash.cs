using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Utilities
{
	using System;

	namespace Opal.Utilities
	{
		public class SimHashGenerator<T>(Func<T, double[]> extractor, int hashBits)
		{
			public Func<T, double[]> Extractor { get; } = extractor;
			public int HashBits { get; } = hashBits;

			public ulong Hash(T item)
			{
				double[] features = Extractor(item);
				int[] vector = new int[HashBits];

				foreach (double feature in features)
				{
					long bits = BitConverter.DoubleToInt64Bits(feature);

					for (int i = 0; i < HashBits; i++)
					{
						int bit = ((bits >> i) & 1) == 1 ? 1 : -1;
						vector[i] += bit;
					}
				}

				ulong hash = 0;
				for (int i = 0; i < HashBits; i++)
				{
					if (vector[i] > 0)
						hash |= (1UL << i);
				}
				return hash;
			}

			public static int HammingDistance(ulong hash1, ulong hash2)
			{
				ulong x = hash1 ^ hash2;
				int dist = 0;
				while (x != 0)
				{
					dist++;
					x &= x - 1;
				}
				return dist;
			}

			public static float HammingSimilarity(ulong hash1, ulong hash2)
			{
				ulong xored = hash1 ^ hash2;
				int differingBits = CountBits(xored);
				return 1.0f - (differingBits / 64f);
			}

			public static int CountBits(ulong value)
			{
				int count = 0;
				while (value != 0)
				{
					value &= (value - 1);
					count++;
				}
				return count;
			}
		}
	}

}
