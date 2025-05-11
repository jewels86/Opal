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

		/// <summary>The percentage of sequences the word must appear in to be considered excessive.</summary>
		public double K { get; set; }

		public ConcurrentDictionary<T, int> WordCount { get; } = new();
		public ConcurrentBag<T> ExcessiveTokens { get; } = new();
		public int TotalSequences { get; private set; } = 0;

		private object _lock = new();

		public ExcessiveUseRecognitionModule(double k, string? name = null)
		{
			ID = Core.Register(this);
			Name = name ?? $"excessive-use-{typeof(T).Name.ToLower()}";
			K = k;
		}

		public void Initialize() { }

		public void Analyze(T[] sequence)
		{
			if (sequence.Length == 0) return;
			foreach (T word in sequence)
			{
				WordCount.AddOrUpdate(word, 1, (key, value) => value + 1);
				if (WordCount[word] > TotalSequences * K)
				{
					ExcessiveTokens.Add(word);
				}
			}
			lock (_lock) { TotalSequences++; }
		}

		public IEnumerable<T> GetExcessiveTokens()
		{
			return ExcessiveTokens.ToArray();
		}

		public bool IsExcessive(T token)
		{
			return ExcessiveTokens.Contains(token);
		}

		public T[] Filter(T[] sequence)
		{
			return sequence.Where(x => !ExcessiveTokens.Contains(x)).ToArray();
		}

		public void Clear()
		{
			WordCount.Clear();
			ExcessiveTokens.Clear();
			TotalSequences = 0;
		}
	}
}
