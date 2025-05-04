using Opal.Utilities;
using Opal.Modules;
using System.Numerics;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			EmbeddingsModule<string> embeddings = new(16, 256, 256, 0.7);
			var e1 = embeddings.CreateEmbedding("hello");
			var hash1 = embeddings.HashGenerator.Hash(e1.Vector);
			Console.WriteLine($"e1: {hash1}");
			var e2 = embeddings.CreateEmbedding("world");
			var hash2 = embeddings.HashGenerator.Hash(e2.Vector);
			Console.WriteLine($"e2: {hash2}");
			for (int i = 0; i < 10; i++)
			{
				embeddings.Associate(e1, e2);
				Console.WriteLine($"cos-similarity {i}: {embeddings.CosineSimilarity(e1.Vector, e2.Vector)}");
				Console.WriteLine($"quick-similarity {i}: {embeddings.QuickSimilarity(e1, e2)}");
			}
		}
	}
}
