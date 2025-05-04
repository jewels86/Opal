using System;
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

		public delegate void NewStorageNodeDelegate(string word);
		public delegate void RemoveStorageNodeDelegate(string word);
		public delegate double GetSimilarityDelegate(string word1, string word2);
		public delegate List<(string, double)> GetSimilarWordsDelegate(string word);

		public NewStorageNodeDelegate NewStorageNode { get; set; }
		public RemoveStorageNodeDelegate RemoveStorageNode { get; set; }
		public GetSimilarityDelegate GetSimilarity { get; set; }
		public GetSimilarWordsDelegate GetSimilarWords { get; set; }

		public SemanticInterpreterModule(NewStorageNodeDelegate? newStorageNode = null, RemoveStorageNodeDelegate? removeStorageNode = null, 
			GetSimilarityDelegate? getSimilarity = null, GetSimilarWordsDelegate? getSimilarWords = null, string? name = null)
		{
			ID = Core.Register(this);
			Name = name ?? "semantic-interpreter";

			if (newStorageNode == null && removeStorageNode == null && getSimilarity == null && getSimilarWords == null)
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
			}
			else
			{
				NewStorageNode = newStorageNode!;
				RemoveStorageNode = removeStorageNode!;
				GetSimilarity = getSimilarity!;
				GetSimilarWords = getSimilarWords!;
			}
		}

		public void Initialize() { }

		public void Receive(Packet packet) { }

		#region Add/Remove Words
		public void AddWord(string word)
		{
			NewStorageNode(word);
		}
		public void RemoveWord(string word)
		{
			RemoveStorageNode(word);
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

	}
}
