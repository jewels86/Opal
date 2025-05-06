using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Opal.Modules;

namespace Opal.Configurations
{
	public static class SemanticInterpreterConfigurations
	{
		public static SemanticInterpreterModule.NewStorageNodeDelegate GenerateNewStorageNodeWithEmbeddings(EmbeddingsModule<string> embeddingsModule)
		{
			return (word) => embeddingsModule.CreateEmbedding(word);
		}
		public static SemanticInterpreterModule.RemoveStorageNodeDelegate GenerateRemoveStorageNodeWithEmbeddings(EmbeddingsModule<string> embeddingsModule)
		{
			return (word) => embeddingsModule.RemoveEmbedding(word);
		}
		public static SemanticInterpreterModule.GetSimilarityDelegate GenerateGetSimilarityWithEmbeddings(EmbeddingsModule<string> embeddingsModule)
		{
			return (word1, word2) =>
			{
				var embedding1 = embeddingsModule.GetEmbedding(word1);
				var embedding2 = embeddingsModule.GetEmbedding(word2);
				if (embedding1 == null || embedding2 == null)
					return double.NaN;
				return embeddingsModule.CosineSimilarity(embedding1.Vector, embedding2.Vector);
			};
		}
		public static SemanticInterpreterModule.GetSimilarWordsDelegate GenerateGetSimilarWordsWithEmbeddings(EmbeddingsModule<string> embeddingsModule)
		{
			return (word) =>
			{
				var embedding = embeddingsModule.GetEmbedding(word);
				if (embedding == null)
					return new List<(string, double)>();
				return [.. embeddingsModule.FindSimilar(embedding).Select(x => (x.Item1.Data, x.Item2))];
			};
		}
		public static SemanticInterpreterModule.AssociateDelegate GenerateAssociateWithEmbeddings(EmbeddingsModule<string> embeddingsModule)
		{
			return (word1, word2, strength) =>
			{
				var embedding1 = embeddingsModule.GetEmbedding(word1);
				var embedding2 = embeddingsModule.GetEmbedding(word2);
				if (embedding1 == null || embedding2 == null)
					return;
				embeddingsModule.Associate(embedding1, embedding2, strength);
			};
		}
	}
}
