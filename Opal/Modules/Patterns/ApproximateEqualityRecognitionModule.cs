using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Modules.Patterns
{
	public class ApproximateEqualityRecognitionModule<T> : IModule where T : notnull
	{
		public string Name { get; set; }
		public int ID { get; set; }

		/// <summary>The threshold for approximate equality. A value between 0 and 1.</summary>
		public double Q { get; set; }
		public Func<T, T> Normalize { get; set; }
		public Func<T, T, bool> IsEqual { get; set; } = (a, b) => a.Equals(b);

		public ApproximateEqualityRecognitionModule(double q, Func<T, T> normalize, Func<T, T, bool>? equality = null, string? name = null)
		{
			Name = name ?? "approximate-equality-recognition";
			ID = Core.Register(this);
			Q = q;
			Normalize = normalize;
			if (equality != null)
			{
				IsEqual = equality;
			}
		}

		public void Initialize() { }

		public bool IsApproximatelyEqual(IEnumerable<T> a, IEnumerable<T> b)
		{
			T[] normalizedA = a.Select(Normalize).ToArray();
			T[] normalizedB = b.Select(Normalize).ToArray();

			int n = normalizedA.Count(), m = normalizedB.Count();
			int[,] dp = new int[n + 1, m + 1];

			for (int i = 0; i <= n; i++) dp[i, 0] = i;
			for (int j = 0; j <= m; j++) dp[0, j] = j;

			for (int i = 1; i <= n; i++)
			{
				for (int j = 1; j <= m; j++)
				{
					int cost = IsEqual(normalizedA[i - 1], normalizedB[j - 1]) ? 0 : 1;
					dp[i, j] = Math.Min(
						Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1),
						dp[i - 1, j - 1] + cost
					);
				}
			}

			int distance = dp[n, m];
			double maxLength = Math.Max(n, m);

			double similarity = 1.0 - (double)distance / maxLength;
			return similarity >= Q;
			// Q: what is a good example for Q?
		}
	}
}
