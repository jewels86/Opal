using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Opal.Utilities.Opal.Utilities;

namespace Opal.Modules
{
	public class EmbeddingsModule<T> : IModule
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

		private int _nextID = -1;
		private object _nextIDLock = new();

		private Random _random = new();

		/// <summary>
		/// Creates a new EmbeddingsModule.
		/// </summary>
		/// <param name="k">The number of buckets as a power of two.</param>
		/// <param name="n">The number of dimensions a single embedding contains.</param>
		/// <param name="h">The number of hash bits to use when SimHashing.</param>
		/// <param name="r">The learning rate to use when associating vectors.</param>
		/// <param name="reduce">A function that takes a ulong hash and converts it to an int bucket.</param>
		public EmbeddingsModule(int k, int n, int h, double r, Func<ulong, int>? reduce = null)
		{
			ID = Core.Register(this);
			Name = $"embeddings-{typeof(T).Name}";
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
			double[] vector = Enumerable.Range(0, N).Select(_ => _random.NextDouble() * 2 - 1).ToArray();
			vector = EmbeddingsModule<T>.Normalize(vector);
			ulong hash = HashGenerator.Hash(vector);
			int bucketID = _reduce(hash);
			int id;
			lock (_nextIDLock) { id = _nextID++; }
			Embedding<T> embedding = new(id, data, vector);

			var bucket = Embeddings.GetOrAdd(bucketID, _ => new());
			lock (bucket) { bucket.Add(embedding); }
			EmbeddingIDs.TryAdd(id, embedding);
			return embedding;
		}
		public bool RemoveEmbedding(Embedding<T> embedding)
		{
			if (!EmbeddingIDs.TryRemove(embedding.ID, out var _))
			{
				return false;
			}
			ulong hash = HashGenerator.Hash(embedding.Vector);
			int bucketID = _reduce(hash);
			if (!Embeddings.TryGetValue(bucketID, out var bucket))
			{
				return false;
			}
			lock (bucket)
			{
				if (bucket.Contains(embedding))
				{
					bucket.Remove(embedding);
					if (bucket.Count == 0)
					{
						Embeddings.TryRemove(bucketID, out var _);
					}
					return true;
				}
			}
			return false;
		}
		public bool RemoveEmbedding(int id)
		{
			if (!EmbeddingIDs.TryRemove(id, out var embedding))
			{
				return false;
			}
			return RemoveEmbedding(embedding);
		}
		#endregion
		#region Associate Embeddings
		public void Associate(Embedding<T> embeddingA, Embedding<T> embeddingB)
		{
			ulong oldHashA = HashGenerator.Hash(embeddingA.Vector);
			ulong oldHashB = HashGenerator.Hash(embeddingB.Vector);

			embeddingA.Vector = Add(Multiply(Subtract(embeddingA.Vector, embeddingB.Vector), R), embeddingA.Vector);
			embeddingB.Vector = Add(Multiply(Subtract(embeddingB.Vector, embeddingA.Vector), R), embeddingB.Vector);
			embeddingA.Vector = Normalize(embeddingA.Vector);
			embeddingB.Vector = Normalize(embeddingB.Vector);

			ulong hashA = HashGenerator.Hash(embeddingA.Vector);
			ulong hashB = HashGenerator.Hash(embeddingB.Vector);
			int bucketIDA = _reduce(hashA);
			int bucketIDB = _reduce(hashB);

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
		}
		#endregion
		#region Get/Find Embedding(s)
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
		#endregion

		#region Similarity
		public double CosineSimilarity(double[] vectorA, double[] vectorB)
		{
			double dotProduct = 0;
			double lengthA = 0;
			double lengthB = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				dotProduct += vectorA[i] * vectorB[i];
				lengthA += vectorA[i] * vectorA[i];
				lengthB += vectorB[i] * vectorB[i];
			}
			if (lengthA == 0 || lengthB == 0)
			{
				return 0;
			}
			return dotProduct / (Math.Sqrt(lengthA) * Math.Sqrt(lengthB));
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
			return SimHashGenerator<double[]>.HammingSimilarity(hashA, hashB);
		}
		public double QuickSimilarity(Embedding<T> embeddingA, Embedding<T> embeddingB)
		{
			return QuickSimilarity(HashGenerator.Hash(embeddingA.Vector), HashGenerator.Hash(embeddingB.Vector));
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
		public static double[] Magnitude(double[] vector)
		{
			double length = Math.Sqrt(vector.Sum(v => v * v));
			return vector.Select(v => v / length).ToArray();
		}
		public static double[] DotProduct(double[] vectorA, double[] vectorB)
		{
			return vectorA.Zip(vectorB, (a, b) => a * b).ToArray();
		}
		#endregion
	}

	public class Embedding<T>(int id, T data, double[] vector)
	{
		public int ID { get; private set; } = id;
		public T Data { get; private set; } = data;
		public double[] Vector { get; set; } = vector;
	}

}
