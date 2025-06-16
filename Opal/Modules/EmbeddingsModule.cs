using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Opal.Utilities.Opal.Utilities;

namespace Opal.Modules
{
	public class EmbeddingsModule<T> : IModule where T : notnull
	{
		public int ID { get; private set; }
		public string Name { get; private set; }

		/// <summary>The number of buckets as a power of two.</summary>
		public int K { get; private set; }
		/// <summary>The number of dimensions a single embedding contains.</summary>
		public int N { get; private set; }
		/// <summary>The number of hash bits to use when SimHashing.</summary>
		public int H { get; private set; }
		/// <summary>The learning rate to use when associating vectors.</summary>
		public double R { get; private set; }
		/// <summary>The embeddings stored in the module (bucketID, embedding[]).</summary>
		public ConcurrentDictionary<int, List<Embedding<T>>> Embeddings { get; private set; } = [];
		/// <summary>The embeddings stored in the module (id, embedding).</summary>
		public ConcurrentDictionary<int, Embedding<T>> EmbeddingIDs { get; private set; } = [];
		public SimHashGenerator<double[]> HashGenerator { get; private set; }

		private Func<ulong, int> _reduce;

		private int _nextID = 0;
		private object _nextIDLock = new();

		private Random _random = new();

		/// <summary>
		/// Creates a new EmbeddingsModule.
		/// </summary>
		/// <param name="k">The number of buckets as a power of two.</param>
		/// <param name="n">The number of dimensions a single embedding contains.</param>
		/// <param name="h">The number of hash bits to use when SimHashing.</param>
		/// <param name="r">The learning rate to use when associating vectors.</param>
		/// <param name="name">Optional name for the module.</param>
		/// <param name="reduce">A function that takes a ulong hash and converts it to an int bucket.</param>
		public EmbeddingsModule(int k, int n, int h, double r, string? name = null, Func<ulong, int>? reduce = null)
		{
			ID = Core.Register(this);
			Name = name ?? $"embeddings-{typeof(T).Name.ToLower()}";
			K = k;
			N = n;
			H = h;
			R = r;
			HashGenerator = new(x => x, H);
			_reduce = reduce ?? (x => (int)(x & (ulong)(K - 1)));
		}

		public void Initialize() { }

		public void Receive(Packet packet)
		{
			var (payload, sender) = packet;

		}

		#region Add/Remove Embedding
		public Embedding<T> CreateEmbedding(T data)
		{
			Core.Log(Name, 2, $"Creating embedding for data: {data} ({typeof(T).Name})");

			double[] vector = Enumerable.Range(0, N).Select(_ => _random.NextDouble() * 2 - 1).ToArray();
			vector = EmbeddingsModule<T>.Normalize(vector);
			ulong hash = HashGenerator.Hash(vector);
			int bucketID = _reduce(hash);
			int id;
			lock (_nextIDLock) { id = _nextID++; }
			Embedding<T> embedding = new(id, data, vector);

			var bucket = Embeddings.GetOrAdd(bucketID, _ => new());
			lock (bucket) { bucket.Add(embedding); }
			EmbeddingIDs[id] = embedding;

			Core.Log(Name, 2, $"Created embedding: {embedding} (with hash {hash})");
			return embedding;
		}
		public bool RemoveEmbedding(Embedding<T> embedding)
		{
			Core.Log(Name, 2, $"Removing embedding: {embedding}");
			int id = embedding.ID;
			if (!EmbeddingIDs.Values.Contains(embedding))
			{
				Core.Log(Name, 3, $"Embedding {embedding} not found in EmbeddingIDs.");
				return false;
			}
			if (!EmbeddingIDs.TryRemove(id, out var _))
			{ 
				Core.Log(Name, 3, $"Failed to remove embedding {embedding} from EmbeddingIDs.");
				return false;
			}
			
			ulong hash = HashGenerator.Hash(embedding.Vector);
			int bucketID = _reduce(hash);
			if (Embeddings.TryGetValue(bucketID, out var bucket))
			{
				lock (bucket)
				{
					if (bucket.Contains(embedding))
					{
						bucket.Remove(embedding);
						if (bucket.Count == 0)
						{
							Embeddings.TryRemove(bucketID, out var _);
						}
					}
				}
			}
			Core.Log(Name, 2, $"Removed embedding: {embedding} (with hash {hash})");
			return true;
		}
		public bool RemoveEmbedding(int id)
		{
			if (!EmbeddingIDs.TryGetValue(id, out Embedding<T>? value))
			{
				return false;
			}
			return RemoveEmbedding(value);
		}
		public bool RemoveEmbedding(T data)
		{
			var embedding = EmbeddingIDs.Values.FirstOrDefault(x => x.Data.Equals(data));
			if (embedding == null)
			{
				return false;
			}
			return RemoveEmbedding(embedding);
		}
		#endregion
		#region Associate Embeddings
		public void Associate(Embedding<T> embeddingA, Embedding<T> embeddingB, double strength)
		{
			Core.Log(Name, 2, $"Associating embeddings: {embeddingA} and {embeddingB} ({typeof(T).Name})");
			ulong oldHashA = HashGenerator.Hash(embeddingA.Vector);
			ulong oldHashB = HashGenerator.Hash(embeddingB.Vector);
			Core.Log(Name, 3, $"Old vectors: {oldHashA} and {oldHashB}");

			embeddingA.Vector = Normalize(Add(Multiply(embeddingA.Vector, 1 - R), Multiply(embeddingB.Vector, R * strength)));
			embeddingB.Vector = Normalize(Add(Multiply(embeddingB.Vector, 1 - R), Multiply(embeddingA.Vector, R * strength)));

			ulong hashA = HashGenerator.Hash(embeddingA.Vector);
			ulong hashB = HashGenerator.Hash(embeddingB.Vector);
			int bucketIDA = _reduce(hashA);
			int bucketIDB = _reduce(hashB);

			Core.Log(Name, 3, $"New vectors: {hashA} and {hashB} (belonging to buckets {bucketIDA} and {bucketIDB} respectively)");

			var bucketA = Embeddings.GetOrAdd(bucketIDA, _ => new());
			var bucketB = Embeddings.GetOrAdd(bucketIDB, _ => new());
			lock (bucketA) { bucketA.Add(embeddingA); }
			lock (bucketB) { bucketB.Add(embeddingB); }

			int oldBucketIDA = _reduce(oldHashA);
			int oldBucketIDB = _reduce(oldHashB);

			if (Embeddings.TryGetValue(oldBucketIDA, out var oldBucketA))
			{
				lock (oldBucketA)
				{
					if (oldBucketA.Contains(embeddingA))
					{
						oldBucketA.Remove(embeddingA);
						if (oldBucketA.Count == 0)
						{
							Embeddings.TryRemove(oldBucketIDA, out var _);
						}
					}
				}
			}
			if (Embeddings.TryGetValue(oldBucketIDB, out var oldBucketB))
			{
				lock (oldBucketB)
				{
					if (oldBucketB.Contains(embeddingB))
					{
						oldBucketB.Remove(embeddingB);
						if (oldBucketB.Count == 0)
						{
							Embeddings.TryRemove(oldBucketIDB, out var _);
						}
					}
				}
			}
			EmbeddingIDs[embeddingA.ID] = embeddingA;
			EmbeddingIDs[embeddingB.ID] = embeddingB;
			Core.Log(Name, 2, $"Associated embeddings: {embeddingA} and {embeddingB} (with hashes {hashA} and {hashB})");
		}
		#endregion
		#region Get Embedding(s)
		public Embedding<T>? GetEmbedding(int id)
		{
			if (EmbeddingIDs.TryGetValue(id, out var embedding))
			{
				return embedding;
			}
			return null;
		}
		public Embedding<T>? GetEmbedding(ulong hash)
		{
			int bucketID = _reduce(hash);
			if (Embeddings.TryGetValue(bucketID, out var bucket))
			{
				return bucket.FirstOrDefault(x => HashGenerator.Hash(x.Vector) == hash);
			}
			return null;
		}
		public Embedding<T>? GetEmbedding(T data)
		{
			return EmbeddingIDs.Values.FirstOrDefault(x => x.Data.Equals(data));
		}
		#endregion
		#region Find Embedding(s)
		public List<(Embedding<T>, double)> FindSimilar(Embedding<T> embedding, int max = 10, double threshold = 0.7, Func<double[], double[], double>? similarityFunction = null)
		{
			Core.Log(Name, 2, $"Finding similar embeddings for: {embedding} ({typeof(T).Name})");
			ulong hash = HashGenerator.Hash(embedding.Vector);
			int originalBucketID = _reduce(hash);

			similarityFunction ??= CosineSimilarity;

			int[] sortedBuckets = [.. Embeddings.Keys.OrderBy(x => Math.Abs(x - originalBucketID))];
			List<(Embedding<T>, double)> results = [];

			foreach (var bucketID in sortedBuckets)
			{
				if (Embeddings.TryGetValue(bucketID, out var bucket))
				{
					bucket.Select(x => (x, similarityFunction(embedding.Vector, x.Vector)))
						.Where(x => x.Item1.ID != embedding.ID)
						.Where(x => x.Item2 >= threshold)
						.OrderByDescending(x => x.Item2)
						.Take(max - results.Count)
						.ToList()
						.ForEach(results.Add);
					if (results.Count >= max)
					{
						results = results.OrderByDescending(x => x.Item2).Take(max).ToList();
						break;
					}
				}
			}
			Core.Log(Name, 2, $"Found {results.Count} similar embeddings for {embedding} (with hash {hash})");
			return results;
		}
		#endregion
		
		# region Static Helpers

		public static Embedding<T> PlaceholderEmbedding(double[] vector) => new(-1, default(T)!, vector);
		# endregion

		#region Similarity
		public double CosineSimilarity(double[] vectorA, double[] vectorB)
		{
			double dotProduct = DotProduct(vectorA, vectorB);
			double magnitudeA = Magnitude(vectorA);
			double magnitudeB = Magnitude(vectorB);
			if (magnitudeA == 0 || magnitudeB == 0)
			{
				return 0;
			}
			return dotProduct / (magnitudeA * magnitudeB);
		}
		public static double PearsonCorrelation(double[] vectorA, double[] vectorB)
		{
			double sumA = vectorA.Sum();
			double sumB = vectorB.Sum();
			double sumASquared = vectorA.Sum(v => v * v);
			double sumBSquared = vectorB.Sum(v => v * v);
			double sumProduct = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sumProduct += vectorA[i] * vectorB[i];
			}
			int n = vectorA.Length;
			double numerator = n * sumProduct - sumA * sumB;
			double denominator = Math.Sqrt((n * sumASquared - sumA * sumA) * (n * sumBSquared - sumB * sumB));
			if (denominator == 0)
			{
				return 0;
			}
			return numerator / denominator;
		}
		public static double EuclideanDistance(double[] vectorA, double[] vectorB)
		{
			double sum = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sum += (vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i]);
			}
			return Math.Sqrt(sum);
		}
		public double QuickSimilarity(ulong hashA, ulong hashB)
		{
			return SimHashGenerator<double[]>.HammingDistance(hashA, hashB);
		}
		public double QuickSimilarity(Embedding<T> embeddingA, Embedding<T> embeddingB)
		{
			return QuickSimilarity(HashGenerator.Hash(embeddingA.Vector), HashGenerator.Hash(embeddingB.Vector));
		}
		public double QuickSimilarity(double[] vectorA, double[] vectorB)
		{
			return QuickSimilarity(HashGenerator.Hash(vectorA), HashGenerator.Hash(vectorB));
		}
		#endregion
		#region Vector Operations
		public static double[] Add(double[] vectorA, double[] vectorB)
		{
			return vectorA.Zip(vectorB, (a, b) => a + b).ToArray();
		}
		public static double[] Subtract(double[] vectorA, double[] vectorB)
		{
			return vectorA.Zip(vectorB, (a, b) => a - b).ToArray();
		}
		public static double[] Multiply(double[] vector, double scalar)
		{
			return vector.Select(v => v * scalar).ToArray();
		}
		public static double[] Average(double[] vectorA, double[] vectorB)
		{
			return vectorA.Zip(vectorB, (a, b) => (a + b) / 2).ToArray();
		}
		public static double[] Normalize(double[] vector)
		{
			double length = Math.Sqrt(vector.Sum(v => v * v));
			if (length == 0)
			{
				return vector;
			}
			return vector.Select(v => v / length).ToArray();
		}
		public static double Magnitude(double[] vector)
		{
			return Math.Sqrt(vector.Sum(v => v * v));
		}
		public static double DotProduct(double[] vectorA, double[] vectorB)
		{
			return vectorA.Zip(vectorB, (a, b) => a * b).Sum();
		}
		#endregion
	}

	public class Embedding<T>(int id, T data, double[] vector)
	{
		public int ID { get; private set; } = id;
		public T Data { get; private set; } = data;
		public double[] Vector { get; set; } = vector;

		public override string ToString()
		{
			return $"Embedding(ID: {ID}, Data: {Data})";
		}
	}

}
