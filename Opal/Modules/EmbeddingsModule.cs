using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Opal.Utilities.Opal.Utilities;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;
using Opal.Utilities;
using static Opal.Utilities.MathFunctions;

namespace Opal.Modules
{
	public class EmbeddingsModule<T> : IModule where T : notnull
	{
		public int ID { get; private set; }
		public string Name { get; private set; }

		/// <summary>The number of buckets as a power of two.</summary>
		public int TotalBuckets { get; private set; }
		/// <summary>The number of dimensions a single embedding contains.</summary>
		public int EmbeddingSize { get; private set; }
		/// <summary>The number of hash bits to use when SimHashing.</summary>
		public int HashBits { get; private set; }
		/// <summary>The learning rate to use when associating vectors.</summary>
		public double LearningRate { get; private set; }
		/// <summary>The embeddings stored in the module (bucketID, embedding[]).</summary>
		public ConcurrentDictionary<int, ConcurrentDictionary<T, Embedding<T>>> Embeddings { get; } = [];
		/// <summary>The embeddings stored in the module (id, embedding).</summary>
		public ConcurrentDictionary<int, Embedding<T>> EmbeddingIDs { get; } = [];
		public SimHashGenerator<double[]> HashGenerator { get; }

		public bool Log { get; set; } = false;

		private Func<ulong, int> _reduce;

		private int _nextID = 0;
		private object _nextIDLock = new();

		private Random _random = new();

		/// <summary>
		/// Creates a new EmbeddingsModule.
		/// </summary>
		/// <param name="totalBuckets">The number of buckets as a power of two.</param>
		/// <param name="embeddingSize">The number of dimensions a single embedding contains.</param>
		/// <param name="hashBits">The number of hash bits to use when SimHashing.</param>
		/// <param name="learningRate">The learning rate to use when associating vectors.</param>
		/// <param name="name">Optional name for the module.</param>
		/// <param name="reduce">A function that takes a ulong hash and converts it to an int bucket.</param>
		public EmbeddingsModule(int totalBuckets, int embeddingSize, int hashBits, double learningRate, string? name = null, Func<ulong, int>? reduce = null)
		{
			ID = Core.Register(this);
			Name = name ?? $"embeddings-{typeof(T).Name.ToLower()}";
			TotalBuckets = totalBuckets;
			EmbeddingSize = embeddingSize;
			HashBits = hashBits;
			LearningRate = learningRate;
			HashGenerator = new(x => x, HashBits);
			_reduce = reduce ?? (x => (int)(x & (ulong)(TotalBuckets - 1)));
		}

		public void Initialize() { }

		#region Add Embeddings
		public Embedding<T> CreateEmbedding(T data)
		{
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Creating embedding for data: {data} ({typeof(T).Name})");

			double[] vector = RandomVector(EmbeddingSize);
			vector = Normalize(vector);
			ulong hash = HashGenerator.Hash(vector);
			int bucketId = _reduce(hash);
			int id;
			lock (_nextIDLock) { id = _nextID++; } // TODO: can we use GUIDs instead?
			Embedding<T> embedding = new(id, data, vector);

			var bucket = Embeddings.GetOrAdd(bucketId, _ => new());
			bucket.AddOrUpdate(data,_ => embedding, (_, _) => embedding);
			EmbeddingIDs[id] = embedding;

			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Created embedding: {embedding} (with hash {hash})");
			return embedding;
		}
		#endregion
		#region Associate Embeddings
		public void Associate(Embedding<T> embeddingA, Embedding<T> embeddingB, double strength)
		{
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Associating embeddings: {embeddingA} and {embeddingB} ({typeof(T).Name})");
			ulong oldHashA = HashGenerator.Hash(embeddingA.Vector);
			ulong oldHashB = HashGenerator.Hash(embeddingB.Vector);
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Old vectors: {oldHashA} and {oldHashB}");

			embeddingA.Vector = Normalize(Add(Multiply(embeddingA.Vector, 1 - LearningRate), Multiply(embeddingB.Vector, LearningRate * strength)));
			embeddingB.Vector = Normalize(Add(Multiply(embeddingB.Vector, 1 - LearningRate), Multiply(embeddingA.Vector, LearningRate * strength)));

			ulong hashA = HashGenerator.Hash(embeddingA.Vector);
			ulong hashB = HashGenerator.Hash(embeddingB.Vector);
			int bucketIdA = _reduce(hashA);
			int bucketIdB = _reduce(hashB);

			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"New vectors: {hashA} and {hashB} (belonging to buckets {bucketIdA} and {bucketIdB} respectively)");

			AddToBucket(bucketIdA, embeddingA);
			AddToBucket(bucketIdB, embeddingB);

			int oldBucketIdA = _reduce(oldHashA);
			int oldBucketIdB = _reduce(oldHashB);

			if (oldBucketIdA != bucketIdA)
			{
				RemoveFromBucket(oldBucketIdA, embeddingA);
			}

			if (oldBucketIdB != bucketIdB)
			{
				RemoveFromBucket(oldBucketIdB, embeddingB);
			}

			EmbeddingIDs[embeddingA.ID] = embeddingA;
			EmbeddingIDs[embeddingB.ID] = embeddingB;
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Associated embeddings: {embeddingA} and {embeddingB} (with hashes {hashA} and {hashB})");
		}
		#endregion
		#region Get Embedding(s)
		public Embedding<T>? GetEmbedding(int id)
		{
			return EmbeddingIDs.GetValueOrDefault(id);
		}
		public Embedding<T>? GetEmbedding(ulong hash)
		{
			int bucketId = _reduce(hash);
			if (Embeddings.TryGetValue(bucketId, out var bucket))
			{
				return bucket.Values.AsParallel().FirstOrDefault(x => HashGenerator.Hash(x.Vector) == hash);
			}
			return null;
		}
		public IEnumerable<T> GetAllData()
		{
			return EmbeddingIDs.Values.AsParallel().Select(x => x.Data);
		}
		public Embedding<T>? GetEmbedding(T data)
		{
			return EmbeddingIDs.Values.AsParallel().FirstOrDefault(x => x.Data.Equals(data));
		}
		#endregion
		#region Find Embedding(s)
		public List<(Embedding<T>, double)> FindSimilar(Embedding<T> embedding, int max = 10, Func<double[], double[], double>? similarityFunction = null)
		{
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Finding closest embeddings for: {embedding} ({typeof(T).Name})");
			ulong hash = HashGenerator.Hash(embedding.Vector);
			int originalBucketId = _reduce(hash);

			similarityFunction ??= CosineSimilarity;

			int[] sortedBuckets = [.. Embeddings.Keys.AsParallel().OrderBy(x => Math.Abs(x - originalBucketId))];
			var allCandidates = new List<(Embedding<T>, double)>();

			foreach (var bucketId in sortedBuckets)
			{
				if (Embeddings.TryGetValue(bucketId, out var bucket))
				{
					allCandidates.AddRange(
						bucket.Values.AsParallel().Select(x => (x, similarityFunction(embedding.Vector, x.Vector)))
						.Where(x => x.Item1.ID != embedding.ID)
					);
				}
			}

			var results = allCandidates.OrderByDescending(x => x.Item2).Take(max).ToList();
			if (Log) Core.Log(Name, Logging.LogLevel.HighDebug, $"Found {results.Count} closest embeddings for {embedding} (with hash {hash})");
			return results;
		}
		#endregion
		
		# region (Static) Helpers
		public void AddToBucket(int bucketId, Embedding<T> embedding)
		{
			var bucket = Embeddings.GetOrAdd(bucketId, _ => new());
			bucket.AddOrUpdate(embedding.Data, _ => embedding, (_, _) => embedding);
			EmbeddingIDs[embedding.ID] = embedding;
		}

		public void RemoveFromBucket(int bucketId, Embedding<T> embedding)
		{
			if (Embeddings.TryGetValue(bucketId, out var bucket))
			{
				bucket.TryRemove(new(embedding.Data, embedding));
			}
		}
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
		#region Save/Load Embeddings
		public void SaveEmbeddingsToFile(string filePath)
		{
			using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
			using var writer = new BinaryWriter(fs);
			writer.Write(EmbeddingIDs.Count);
			writer.Write(TotalBuckets);
			writer.Write(EmbeddingSize);
			writer.Write(HashBits);
			writer.Write(LearningRate);
			foreach (var e in EmbeddingIDs.Values)
			{
				writer.Write(e.ID);
				// For T: if it's a primitive or string, write directly; else, use ToString()
				if (typeof(T) == typeof(string))
					writer.Write((string)(object)e.Data!);
				else if (typeof(T).IsPrimitive)
					writer.Write(Convert.ToString(e.Data) ?? "");
				else
					writer.Write(e.Data?.ToString() ?? "");
				writer.Write(e.Vector.Length);
				for (int i = 0; i < e.Vector.Length; i++)
					writer.Write(e.Vector[i]);
			}
			Core.Log(Name, Logging.LogLevel.LowInfo, $"Saved {EmbeddingIDs.Count} embeddings to {filePath}");
		}
		public void LoadEmbeddingsFromFile(string filePath)
		{
			if (!File.Exists(filePath))
			{
				Core.Log(Name, Logging.LogLevel.LowWarning, $"File not found: {filePath}");
				return;
			}
			using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
			using var reader = new BinaryReader(fs);
			int count = reader.ReadInt32();
			TotalBuckets = reader.ReadInt32();
			EmbeddingSize = reader.ReadInt32();
			HashBits = reader.ReadInt32();
			LearningRate = reader.ReadDouble();
			EmbeddingIDs.Clear();
			Embeddings.Clear();
			for (int i = 0; i < count; i++)
			{
				int id = reader.ReadInt32();
				T data;
				if (typeof(T) == typeof(string))
					data = (T)(object)reader.ReadString();
				else if (typeof(T).IsPrimitive)
					data = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
				else
					data = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
				int len = reader.ReadInt32();
				double[] vector = new double[len];
				for (int j = 0; j < len; j++)
					vector[j] = reader.ReadDouble();
				var embedding = new Embedding<T>(id, data, vector);
				EmbeddingIDs[embedding.ID] = embedding;
				ulong hash = HashGenerator.Hash(embedding.Vector);
				int bucketID = _reduce(hash);
				AddToBucket(bucketID, embedding);
			}
			Core.Log(Name, Logging.LogLevel.LowInfo, $"Loaded {count} embeddings from {filePath}");
		}

		private class SerializableEmbedding<TData>
		{
			public int ID { get; set; }
			public TData Data { get; set; } = default!;
			public double[] Vector { get; set; } = default!;
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
