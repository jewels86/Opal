using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Opal.Modules
{
	public class SemanticInterpreterModule : IModule
	{
		public int ID { get; private set; }
		public string Name { get; private set; }

		/// <summary>The learning rate for the semantic interpreter.</summary>
		public double L { get; set; } = 0.2;

		public delegate void NewStorageNodeDelegate(string word);
		public delegate void RemoveStorageNodeDelegate(string word);
		public delegate double GetSimilarityDelegate(string word1, string word2);
		public delegate List<(string, double)> GetSimilarWordsDelegate(string word);
		public delegate void AssociateDelegate(string word1, string word2, double strength);

		public NewStorageNodeDelegate NewStorageNode { get; set; }
		public RemoveStorageNodeDelegate RemoveStorageNode { get; set; }
		public GetSimilarityDelegate GetSimilarity { get; set; }
		public GetSimilarWordsDelegate GetSimilarWords { get; set; }
		public AssociateDelegate Associate { get; set; }

		public HashSet<string> Added { get; private set; } = [];
		public ConcurrentDictionary<string, SortedDictionary<string, int>> WordTransitions { get; private set; } = [];

		private object _addLock = new();
		private object _sortedDictionaryLock = new();

		public SemanticInterpreterModule(NewStorageNodeDelegate? newStorageNode = null, RemoveStorageNodeDelegate? removeStorageNode = null, 
			GetSimilarityDelegate? getSimilarity = null, GetSimilarWordsDelegate? getSimilarWords = null,
			AssociateDelegate? associate = null, string? name = null)
		{
			ID = Core.Register(this);
			Name = name ?? "semantic-interpreter";

			if (newStorageNode == null || removeStorageNode == null || getSimilarity == null || getSimilarWords == null || associate == null)
			{
				EmbeddingsModule<string> embeddingsModule = new(32, 256, 256, 0.5);
				Core.Register(embeddingsModule);
				NewStorageNode = (word) => embeddingsModule.CreateEmbedding(word);
				RemoveStorageNode = (word) => embeddingsModule.RemoveEmbedding(word);
				GetSimilarity = (word1, word2) =>
				{
					var embedding1 = embeddingsModule.GetEmbedding(word1);
					var embedding2 = embeddingsModule.GetEmbedding(word2);
					if (embedding1 == null || embedding2 == null)
						return double.NaN;
					return embeddingsModule.CosineSimilarity(embedding1.Vector, embedding2.Vector);
				};
				GetSimilarWords = (word) =>
				{
					var embedding = embeddingsModule.GetEmbedding(word);
					if (embedding == null)
						return new List<(string, double)>();
					return [.. embeddingsModule.FindSimilar(embedding).Select(x => (x.Item1.Data, x.Item2))];
				};
				Associate = (word1, word2, strength) =>
				{
					var embedding1 = embeddingsModule.GetEmbedding(word1);
					var embedding2 = embeddingsModule.GetEmbedding(word2);
					if (embedding1 == null || embedding2 == null)
						return;
					embeddingsModule.Associate(embedding1, embedding2, strength);
				};
			}
			else
			{
				NewStorageNode = newStorageNode!;
				RemoveStorageNode = removeStorageNode!;
				GetSimilarity = getSimilarity!;
				GetSimilarWords = getSimilarWords!;
				Associate = associate!;
			}
		}

		public void Initialize() { }

		public void Receive(Packet packet) { }

		#region Add/Remove Words
		public void AddWord(string word)
		{
			NewStorageNode(word);
			lock (_addLock) { Added.Add(word); }
		}
		public void RemoveWord(string word)
		{
			RemoveStorageNode(word);
			lock (_addLock) { Added.Remove(word); }
		}
		#endregion
		#region Similarity
		public double GetSimilarityBetween(string word1, string word2)
		{
			return GetSimilarity(word1, word2);
		}
		public List<(string, double)> GetSimilar(string word)
		{
			return GetSimilarWords(word);
		}
		public List<(string, double)> GetSimilar(string word, int count)
		{
			return [.. GetSimilarWords(word).OrderByDescending(x => x.Item2).Take(count)];
		}
		public List<(string, double)> GetSimilar(string word, int count, double threshold)
		{
			return [.. GetSimilarWords(word).Where(x => x.Item2 >= threshold).OrderByDescending(x => x.Item2).Take(count)];
		}
		public List<(string, double)> GetSimilar(string word, double threshold)
		{
			return [.. GetSimilarWords(word).Where(x => x.Item2 >= threshold).OrderByDescending(x => x.Item2)];
		}
		#endregion
		#region Interpret
		public void Interpret(string[] sentence)
		{
			sentence = sentence.Prepend("[special-start]").Append("[special-end]").ToArray();
			Core.Log(Name, 2, "Interpreting sentence: " + string.Join(" ", sentence));

			foreach (var word in sentence)
			{
				if (!Added.Contains(word))
				{
					Core.Log(Name, 3, "Adding word: " + word);
					AddWord(word);
				}
			}

			for (int i = 0; i < sentence.Length; i++)
			{
				for (int j = 0; j < sentence.Length; j++)
				{
					if (i == j)
						continue;
					var distance = 1/(i-j);
					if (distance < 0) Associate(sentence[i], sentence[j], distance * L * 0.5);
					else Associate(sentence[j], sentence[i], distance * L);
				}

				if (i == sentence.Length - 1)
					continue;
				string currentWord = sentence[i];
				string nextWord = sentence[i + 1];

				WordTransitions.AddOrUpdate(currentWord,
				_ =>
				{
					SortedDictionary<string, int> sortedDictionary = new SortedDictionary<string, int>();
					lock (_sortedDictionaryLock)
					{
						sortedDictionary[nextWord] = 1;
					}
					return sortedDictionary;
				},
				(_, existingSortedDictionary) =>
				{
					lock (_sortedDictionaryLock)
					{
						if (existingSortedDictionary.ContainsKey(nextWord))
							existingSortedDictionary[nextWord]++;
						else
							existingSortedDictionary[nextWord] = 1;
					}
					return existingSortedDictionary;
				});
			}

			Core.Log(Name, 2, "Finished interpreting sentence: " + string.Join(" ", sentence));
		}
		#endregion
		#region Next Word
		public List<(string, int)> NextWords(string word, int count)
		{
			if (!WordTransitions.ContainsKey(word))
				return [];
			var sortedDictionary = WordTransitions[word];
			return [.. sortedDictionary.OrderByDescending(x => x.Value).Take(count).Select(x => (x.Key, x.Value))];
		}
		#endregion
	}
}
