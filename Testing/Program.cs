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

			// Test: Create embeddings
			var embedding1 = embeddings.CreateEmbedding("Data1");
			var embedding2 = embeddings.CreateEmbedding("Data2");
			Console.WriteLine($"Created Embedding1 ID: {embedding1.ID}, Data: {embedding1.Data}");
			Console.WriteLine($"Created Embedding2 ID: {embedding2.ID}, Data: {embedding2.Data}");

			var similarity = embeddings.CosineSimilarity(embedding1.Vector, embedding2.Vector);
			Console.WriteLine($"Cosine Similarity between Embedding1 and Embedding2: {similarity}");

			// Test: Retrieve embeddings by ID
			var retrievedEmbedding = embeddings.GetEmbedding(embedding1.ID);
			Console.WriteLine($"Retrieved Embedding ID: {retrievedEmbedding?.ID}, Data: {retrievedEmbedding?.Data}");

			// Test: Associate embeddings
			embeddings.Associate(embedding1, embedding2);
			Console.WriteLine("Associated Embedding1 and Embedding2.");

			similarity = embeddings.CosineSimilarity(embedding1.Vector, embedding2.Vector);
			Console.WriteLine($"Cosine Similarity after association: {similarity}");

			// Test: Find similar embeddings
			var similarEmbeddings = embeddings.FindSimilar(embedding1, max: 5, threshold: 0.5);
			Console.WriteLine($"Found {similarEmbeddings.Count} similar embeddings for Embedding1: {string.Join(", ", similarEmbeddings)}.");

			foreach (var (id, embedding) in embeddings.EmbeddingIDs)
			{
				Console.WriteLine($"Embedding ID: {id}, Data: {embedding.Data}");
			}

			// Test: Remove embeddings
			bool removed = embeddings.RemoveEmbedding(embedding1.ID);
			Console.WriteLine($"Removed Embedding1: {removed}");

			// Test: Attempt to retrieve removed embedding
			var removedEmbedding = embeddings.GetEmbedding(embedding1.ID);
			Console.WriteLine($"Retrieved Removed Embedding: {removedEmbedding?.ID ?? -1}");
		}
	}
}
