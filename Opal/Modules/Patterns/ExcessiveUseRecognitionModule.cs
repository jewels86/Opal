using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Opal.Modules.Patterns
{
	public class ExcessiveUseRecognitionModule<T> : IModule where T : notnull
	{
		public int ID { get; private set; }
		public string Name { get; private set; }

		/// <summary>
		/// Controls how much more frequent than average a token must be to be considered excessive.
		/// </summary>
		public double K { get; set; }

		private long _totalCount = 0;
		public long TotalCount => Interlocked.Read(ref _totalCount);

		private ConcurrentDictionary<T, int> Tokens { get; } = new();

		public double AverageFrequency
		{
			get
			{
				int tokenCount = Tokens.Count;
				return tokenCount == 0 ? 0 : (double)TotalCount / tokenCount;
			}
		}

		public double Threshold => K * AverageFrequency;

		public ExcessiveUseRecognitionModule(double k, string? name = null)
		{
			ID = Core.Register(this);
			Name = name ?? "excessive-use-recognition";
			K = k;
		}

		public void Initialize() { }

		public void Analyze(T[] data)
		{
			Interlocked.Add(ref _totalCount, data.Length);

			data.AsParallel().ForAll(item =>
			{
				Tokens.AddOrUpdate(item, 1, (key, oldValue) => oldValue + 1);
			});
		}

		public bool IsExcessive(T token)
		{
			if (Tokens.TryGetValue(token, out int count))
			{
				return count > Threshold;
			}
			return false;
		}

		public T[] Filter(T[] data)
		{
			return data.Where(x => !IsExcessive(x)).ToArray();
		}
	}
}
