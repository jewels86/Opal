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

			// Create a 4x4 grid of embeddings
			var grid = new Embedding<string>[4, 4];
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					grid[i, j] = embeddings.CreateEmbedding($"cell_{i}_{j}");
					var hash = embeddings.HashGenerator.Hash(grid[i, j].Vector);
					Console.WriteLine($"Embedding at ({i},{j}): {hash}");
				}
			}

			// Associate some embeddings
			embeddings.Associate(grid[0, 0], grid[1, 1]);
			embeddings.Associate(grid[2, 2], grid[3, 3]);
			embeddings.Associate(grid[0, 3], grid[3, 0]);

			// Display cosine similarity for associated embeddings
			Console.WriteLine($"similarity (0,0) and (1,1): {embeddings.CosineSimilarity(grid[0, 0].Vector, grid[1, 1].Vector)}");
			Console.WriteLine($"similarity (2,2) and (3,3): {embeddings.CosineSimilarity(grid[2, 2].Vector, grid[3, 3].Vector)}");
			Console.WriteLine($"similarity (0,3) and (3,0): {embeddings.CosineSimilarity(grid[0, 3].Vector, grid[3, 0].Vector)}");

			// Fix for CS1501: Use string.Join with two arguments
			var similarEmbeddings = embeddings.FindSimilar(grid[0, 0], 1);
			Console.WriteLine($"Found similarity for (0,0): {string.Join(", ", similarEmbeddings)}");
		}
	}
}
