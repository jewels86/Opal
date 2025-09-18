using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Opal.Utilities;
using Opal.Utilities.Concurrency;
using static Opal.Utilities.Logging;
using static Opal.Utilities.Logging.LogLevel;
using static Opal.Utilities.Logging.AddedLogLevel;

namespace Opal.Modules
{
	public class SemanticInterpreterModule<TType, TMemory> : IModule where TType : notnull where TMemory : ISemanticInterpreterMemoryModule<TType>
	{
		public string Name { get; }
		public Logging.LogLevel Baseline { get; set; } = Logging.LogLevel.Info;
		public bool LoggingEnabled { get; set; } = true;

		public TMemory Memory { get; }
		public ConcurrentHashSet<TType> Added { get; set; } = [];
		public TType SpecialStart { get; }
		public TType SpecialEnd { get; }

		public SemanticInterpreterModule(TMemory memory, TType specialStart, TType specialEnd, string? name = null)
		{
			Memory = memory;
			Name = name ?? $"semantic-interpreter-{typeof(TType).Name.ToLower()} with {Memory.Name} memory";
			SpecialStart = specialStart;
			SpecialEnd = specialEnd;
		}

		#region Add/Remove Words
		public void AddWord(TType word)
		{
			Memory.NewStorageNode(word);
			Added.Add(word);
		}
		public void RemoveWord(TType word) => Added.Remove(word);
		#endregion
		#region Similarity
		public double GetSimilarityBetween(TType word1, TType word2) => Memory.GetSimilarity(word1, word2);
		public List<(TType word, double similarity)> GetSimilar(TType word, int? count = null, double? threshold = null) 
			=> Memory.GetSimilarWords(word, count, threshold);
		#endregion
		#region Interpret
		public void Interpret(TType[] sentence, bool parallel = false)
		{
			sentence = sentence.Prepend(SpecialStart).Append(SpecialEnd).ToArray();
			Log(Name, Baseline.Add(LowBaseline), "Interpreting sentence: " + string.Join(" ", sentence));

			sentence.ForAllParallel(word =>
			{
				if (Added.Contains(word)) return;
				Log(Name, Baseline.Add(Unimportant), "Adding word: " + word);
				AddWord(word);
			}, parallel);

			for (int i = 0; i < sentence.Length; i++)
			{
				for (int j = 0; j < sentence.Length; j++)
				{
					if (i == j)
						continue;
					var distance = 1 / (i - j);
					if (distance < 0) Memory.Associate(sentence[i], sentence[j], distance * 0.5);
					else Memory.Associate(sentence[j], sentence[i], distance);
				}
			}

			Log(Name, Baseline, "Finished interpreting sentence: " + string.Join(" ", sentence));
		}
		#endregion
	}

	public interface ISemanticInterpreterMemoryModule<TType> : IModule where TType : notnull
	{
		public void NewStorageNode(TType word);
		public double GetSimilarity(TType word1, TType word2);
		public List<(TType, double)> GetSimilarWords(TType word, int? max, double? threshold);
		public void Associate(TType from, TType to, double weight);
	}
}
