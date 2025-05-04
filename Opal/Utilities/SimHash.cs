using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Utilities
{
	public class SimHash<T>(Func<T, int[]> extractor, int hashBits)
	{
		public Func<T, int[]> Extractor { get; } = extractor;
		public int HashBits { get; } = hashBits;

		public ulong Hash(T item)
		{
			int[] features = Extractor(item);
			int[] vector = new int[HashBits];

			foreach (int feature in features)
			{
				for (int i = 0; i < HashBits; i++)
				{
					int bit = ((feature >> i) & 1) == 1 ? 1 : -1;
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
	}
}
