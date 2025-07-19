using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Opal.Modules.Patterns
{
	public enum ExcessiveUseRecognitionModuleMode
	{
		Percentage, Threshold
	}
	public class ExcessiveUseRecognitionModule<T> : IModule, IAnalyzer<T> where T : notnull
	{
		public int ID { get; private set; }
		public string Name { get; private set; }
		
		/// <summary>
		/// If `Mode` is set to percentage: The percentage of sequences the word must appear in to be considered excessive.
		/// If `Mode` is set to threshold: The minimum number of sequences the word must appear in to be considered excessive.
		/// </summary>
		public double K { get; set; }

		public ExcessiveUseRecognitionModuleMode Mode { get; set; } = ExcessiveUseRecognitionModuleMode.Percentage;

		public ConcurrentDictionary<T, int> WordCount { get; } = new();
		public ConcurrentBag<T> ExcessiveTokens { get; } = new();
		public int TotalSequences { get; private set; } = 0;
		public Func<T, T> Normalizer { get; set; }

		private object _lock = new();

		public List<T[]> Sequences { get; } = new();

		public ExcessiveUseRecognitionModule(double k, Func<T, T>? normalizer = null, string? name = null)
		{
			ID = Core.Register(this);
			Name = name ?? $"excessive-use-{typeof(T).Name.ToLower()}";
			K = k;
			Normalizer = normalizer ?? (x => x);
		}

		public void Initialize() { }

		public void Analyze(IEnumerable<T> sequence)
		{
			if (sequence.Any()) return;
			lock (_lock)
			{
				Sequences.Add(sequence.ToArray());
			}
		}

		public void FinalizeAnalysis()
		{
			WordCount.Clear();
			ExcessiveTokens.Clear();
			TotalSequences = Sequences.Count;
			var normalizedSequences = Sequences.Select(seq => seq.Select(Normalizer).Distinct());
			foreach (var normalizedWords in normalizedSequences)
			{
				foreach (T word in normalizedWords)
				{
					WordCount.AddOrUpdate(word, 1, (key, value) => value + 1);
				}
			}
			foreach (var kvp in WordCount)
			{
				if (Mode == ExcessiveUseRecognitionModuleMode.Percentage)
				{
					if (kvp.Value / (double)TotalSequences >= K)
					{
						ExcessiveTokens.Add(kvp.Key);
					}
				}
				else if (Mode == ExcessiveUseRecognitionModuleMode.Threshold)
				{
					if (kvp.Value >= K)
					{
						ExcessiveTokens.Add(kvp.Key);
					}
				}
			}
		}

		public IEnumerable<T> Results()
		{
			return ExcessiveTokens.ToArray();
		}

		public bool IsExcessive(T token)
		{
			return ExcessiveTokens.Contains(Normalizer(token));
		}

		public T[] Filter(T[] sequence)
		{
			return sequence.Select(Normalizer).Where(x => !ExcessiveTokens.Contains(x)).ToArray();
		}

		public void Clear()
		{
			WordCount.Clear();
			ExcessiveTokens.Clear();
			TotalSequences = 0;
			Sequences.Clear();
		}
	}
}
