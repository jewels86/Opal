using System.Collections.Concurrent;
using Opal.Utilities.Opal.Utilities;
using Opal.Utilities;
using static Opal.Utilities.MathFunctions;
using static Opal.Utilities.Logging.LogLevel;
using static Opal.Utilities.Logging.AddedLogLevel;
using static Opal.Utilities.Logging;
using Opal.Utilities.Concurrency;

namespace Opal.Modules
{
	public class EmbeddingsModule<T> : IModule where T : notnull
	{
		public string Name { get; }

		/// <summary>The number of buckets as a power of two.</summary>
		public int TotalBuckets { get; private set; }
		/// <summary>The number of dimensions a single embedding contains.</summary>
		public int EmbeddingSize { get; private set; }
		/// <summary>The number of hash bits to use when SimHashing.</summary>
		public int HashBits { get; private set; }
		/// <summary>The learning rate to use when associating vectors.</summary>
		public double LearningRate { get; private set; }
		/// <summary>The embeddings stored in the module (bucketID, embedding[]).</summary>
		public ConcurrentDictionary<int, ConcurrentDictionary<Guid, Embedding<T>>> Embeddings { get; } = [];
		/// <summary>The embeddings stored in the module (id, embedding).</summary>
		public ConcurrentDictionary<Guid, Embedding<T>> EmbeddingIDs { get; } = [];
		/// <summary>The embeddings stored in the module (data, embedding).</summary>
		public ConcurrentDictionary<T, Embedding<T>> EmbeddingData { get; } = [];
		/// <summary>The SimHash generator used to hash embeddings.</summary>
		public SimHashGenerator<double[]> HashGenerator { get; }

		public bool LoggingEnabled { get; set; } = false;

		public LogLevel Baseline { get; set; } = LowDebug;

		readonly private Func<ulong, int> _reduce;

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
			Name = name ?? $"embeddings-{typeof(T).Name.ToLower()}";
			TotalBuckets = totalBuckets;
			EmbeddingSize = embeddingSize;
			HashBits = hashBits;
			LearningRate = learningRate;
			HashGenerator = new(x => x, HashBits);
			_reduce = reduce ?? (x => (int)(x & (ulong)(TotalBuckets - 1)));
		}

		#region Add/Remove Embeddings
		public Embedding<T> CreateEmbedding(T data)
		{
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Creating embedding for data: {data} ({typeof(T).Name})");

			double[] vector = RandomVector(EmbeddingSize);
			vector = Normalize(vector);
			ulong hash = HashGenerator.Hash(vector);
			int bucketId = _reduce(hash);
			Guid id = Guid.NewGuid();
			Embedding<T> embedding = new(id, data, vector);

			AddToBucket(bucketId, embedding);
			EmbeddingIDs[id] = embedding;
			EmbeddingData[data] = embedding;

			if (LoggingEnabled) Log(Name, Baseline, $"Created embedding: {embedding} (with hash {hash})");
			return embedding;
		}
		public bool RemoveEmbedding(Guid id)
		{
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Removing embedding with ID: {id} ({typeof(T).Name})");
			if (EmbeddingIDs.TryRemove(id, out var embedding) && EmbeddingData.TryRemove(embedding.Data, out _))
			{
				ulong hash = HashGenerator.Hash(embedding.Vector);
				int bucketId = _reduce(hash);
				RemoveFromBucket(bucketId, embedding);
				if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Removed embedding: {embedding} (with hash {hash})");
				return true;
			}
			if (LoggingEnabled) Log(Name, Baseline, $"Failed to remove embedding with ID: {id} (not found)");
			return false;
		}
		public bool RemoveEmbedding(T data)
		{
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Removing embedding with data: {data} ({typeof(T).Name})");
			if (EmbeddingData.TryRemove(data, out var embedding) && EmbeddingIDs.TryRemove(embedding.Id, out _))
			{
				ulong hash = HashGenerator.Hash(embedding.Vector);
				int bucketId = _reduce(hash);
				RemoveFromBucket(bucketId, embedding);
				if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Removed embedding: {embedding} (with hash {hash})");
				return true;
			}
			if (LoggingEnabled) Log(Name, Baseline, $"Failed to remove embedding with data: {data} (not found)");
			return false;
		}
		#endregion
		#region Associate Embeddings
		public void Associate(Embedding<T> embeddingA, Embedding<T> embeddingB, double strength)
		{
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Associating embeddings: {embeddingA} and {embeddingB} ({typeof(T).Name})");
			ulong oldHashA = HashGenerator.Hash(embeddingA.Vector);
			ulong oldHashB = HashGenerator.Hash(embeddingB.Vector);
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Old vectors: {oldHashA} and {oldHashB}");

			double[] newAVector = Normalize(Add(Multiply(embeddingA.Vector, 1 - LearningRate), Multiply(embeddingB.Vector, LearningRate * strength)));
			double[] newBVector = Normalize(Add(Multiply(embeddingB.Vector, 1 - LearningRate), Multiply(embeddingA.Vector, LearningRate * strength)));
			embeddingA = embeddingA with { Vector = newAVector };
			embeddingB = embeddingB with { Vector = newBVector };
			
			ulong hashA = HashGenerator.Hash(embeddingA.Vector);
			ulong hashB = HashGenerator.Hash(embeddingB.Vector);
			int bucketIdA = _reduce(hashA);
			int bucketIdB = _reduce(hashB);

			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"New vectors: {hashA} and {hashB} (belonging to buckets {bucketIdA} and {bucketIdB} respectively)");

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

			EmbeddingIDs[embeddingA.Id] = embeddingA;
			EmbeddingIDs[embeddingB.Id] = embeddingB;
			if (LoggingEnabled) Log(Name, Baseline, $"Associated embeddings: {embeddingA} and {embeddingB} (with hashes {oldHashA} -> {hashA} and {oldHashB} -> {hashB})");
		}
		#endregion
		#region Get Embedding(s)
		public Embedding<T>? GetEmbedding(Guid id) => EmbeddingIDs.GetValueOrDefault(id);
		/// <summary>
		/// Gets an embedding by its hash. Note that this operation may return null if the hash does not exist.
		/// </summary>
		/// <param name="hash">The hash of the target embedding.</param>
		/// <returns>The target embedding or null if not found.</returns>
		/// <remarks>
		/// This is an EXPENSIVE operation as it requires searching through all embeddings in the bucket.
		/// It's recommended to use GetEmbedding(Guid id) or GetEmbedding(T data) instead.
		/// </remarks>
		public Embedding<T>? GetEmbedding(ulong hash) => Embeddings.GetValueOrDefault(_reduce(hash))?.Values.AsParallel().FirstOrDefault(x => HashGenerator.Hash(x.Vector) == hash);
		/// <summary>
		/// Gets all embeddings stored in the module.
		/// </summary>
		/// <returns>All the embeddings in <see cref="EmbeddingIDs"/></returns>
		/// <remarks>
		/// This is an EXPENSIVE operation as it requires iterating through all embeddings in the module.
		/// It's recommended to search for specific embeddings or at least buckets;
		/// however, this method will search faster than iterating through the dictionary manually.
		/// </remarks>
		public IEnumerable<T> GetAllData() => EmbeddingIDs.Values.AsParallel().Select(x => x.Data);
		public Embedding<T>? GetEmbedding(T data) => EmbeddingData.GetValueOrDefault(data);
		#endregion
		#region Find Embedding(s)
		public List<(Embedding<T>, double)> FindSimilar(Embedding<T> embedding, int max = 10, int bucketsToSearch = -1, 
			Func<double[], double[], double>? similarityFunction = null, bool parallel = true)
		{
			if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), $"Finding closest embeddings for: {embedding} ({typeof(T).Name})");
			ulong hash = HashGenerator.Hash(embedding.Vector);
			int originalBucketId = _reduce(hash);

			similarityFunction ??= CosineSimilarity;

			int[] sortedBuckets = [.. Embeddings.Keys.OrderBy(x => Math.Abs(x - originalBucketId))];
			if (bucketsToSearch != -1) sortedBuckets = sortedBuckets.Take(bucketsToSearch).ToArray();
			var allCandidates = new List<(Embedding<T>, double)>();

			foreach (var bucketId in sortedBuckets)
			{
				if (allCandidates.Count >= max) break;
				if (Embeddings.TryGetValue(bucketId, out var bucket))
				{
					var candidates = bucket.Values.AsParallel(parallel)
						.Where(x => x.Id != embedding.Id)
						.Select(x => (x, similarityFunction(embedding.Vector, x.Vector)))
						.OrderByDescending(x => x.Item2)
						.Take(max - allCandidates.Count)
						.ToList();
					allCandidates.AddRange(candidates);
				}
			}

			var results = allCandidates
				.OrderByDescending(x => x.Item2)
				.Take(max)
				.ToList();
			if (LoggingEnabled) Log(Name, Baseline, $"Found {results.Count} closest embeddings for {embedding} (with hash {hash})");
			return results;
		}
		#endregion
		
		# region (Static) Helpers
		public void AddToBucket(int bucketId, Embedding<T> embedding)
		{
			var bucket = Embeddings.GetOrAdd(bucketId, _ => new());
			bucket.AddOrUpdate(embedding.Id, _ => embedding, (_, _) => embedding);
			EmbeddingIDs[embedding.Id] = embedding;
		}

		public void RemoveFromBucket(int bucketId, Embedding<T> embedding)
		{
			if (Embeddings.TryGetValue(bucketId, out var bucket))
			{
				bucket.TryRemove(new(embedding.Id, embedding));
			}
		}
		public static Embedding<T> PlaceholderEmbedding(double[] vector) => new(Guid.Empty, default(T)!, vector);
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
				writer.Write(e.Id.ToString());
				if (typeof(T) == typeof(string))
					writer.Write((string)(object)e.Data);
				else if (typeof(T).IsPrimitive)
					writer.Write(Convert.ToString(e.Data) ?? "");
				else
					writer.Write(e.Data.ToString() ?? "");
				writer.Write(e.Vector.Length);
				foreach (var v in e.Vector)
					writer.Write(v);
			}
			Log(Name, Baseline.Add(HighBaseline), $"Saved {EmbeddingIDs.Count} embeddings to {filePath}");
		}
		public void LoadEmbeddingsFromFile(string filePath)
		{
			if (!File.Exists(filePath))
			{
				Log(Name, Baseline.Add(HighBaseline), $"File not found: {filePath}");
				throw new FileNotFoundException($"File not found: {filePath}");
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
				Guid id = Guid.Parse(reader.ReadString());
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
				EmbeddingIDs[embedding.Id] = embedding;
				ulong hash = HashGenerator.Hash(embedding.Vector);
				int bucketId = _reduce(hash);
				AddToBucket(bucketId, embedding);
			} 
			Log(Name, Baseline, $"Loaded {count} embeddings from {filePath}");
		}
		#endregion
	}

	public readonly record struct Embedding<T>(Guid Id, T Data, double[] Vector) where T : notnull
	{
		public bool Equals(Embedding<T> other) =>
			Id.Equals(other.Id)
			&& EqualityComparer<T>.Default.Equals(Data, other.Data)
			&& MathFunctions.Equals(Vector, other.Vector);

		public override int GetHashCode()
		{
			int hash = HashCode.Combine(Id, Data);
			foreach (var v in Vector)
				hash = HashCode.Combine(hash, v);
			return hash;
		}

		public override string ToString() => $"Embedding(ID: {Id}, Data: {Data})";
	}


}
