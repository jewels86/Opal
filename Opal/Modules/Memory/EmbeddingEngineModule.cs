using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Memory
{
	[MessagePackObject]
	public class EmbeddingNode
	{
		[Key(0)]
		public float[] Vector { get; set; } = new float[128];
		[Key(1)]
		public Dictionary<string, (float[], float)> Associations { get; set; } = new();
		[Key(2)]
		public Dictionary<string, float> Metadata { get; set; } = new();
	}

	public class EmbeddingEngineModule : IModule
	{
		public string ID => "memory:embedding-engine";
		public List<MemoryStream> Inputs { get; } = [new()];
		public List<object> InputLocks { get; } = [new()];
		public List<bool> Available { get; } = [true];

		public string Type { get; } = "null";

		public int EmbeddingSize { get; } = 128;
		public List<EmbeddingNode> Nodes { get; } = new();

		private Random _random = new();

		public void Initialize(Context ctx)
		{
			ctx.Log(ID, 3, "Initializing EmbeddingEngineModule.");
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			ctx.Log(ID, 3, "Starting main loop of EmbeddingEngineModule.");

			Action<int> main = (i) =>
			{
				Span<byte> bytes = new();
				lock (InputLocks[i]) { Inputs[i].Read(bytes); }
				Packet packet = MessagePackSerializer.Deserialize<Packet>(bytes.ToArray());

				ctx.Log(ID, 3, $"Received packet of type '{packet.Type}' from source '{packet.SourceID}'.");

				if (packet.Type == "memory:embedding-engine->create")
				{
					float[] vector = new float[EmbeddingSize];
					for (int j = 0; j < EmbeddingSize; j++)
					{
						vector[j] = (float)_random.NextDouble();
					}
					EmbeddingNode node = new();
					node.Vector = vector;
					Nodes.Add(node);

					ctx.Log(ID, 3, $"Created new embedding node with vector hash '{SHAHash(vector)}'.");

					Packet response = new()
					{
						TargetID = packet.SourceID,
						SourceID = ID,
						Type = "memory:embedding-engine->create-response",
						PayloadType = "node",
						Payload = MessagePackSerializer.Serialize(node)
					};
					ctx.Send(response);
				}
				else if (packet.Type == "memory:embedding-engine->associate")
				{
					float[] vector = MessagePackSerializer.Deserialize<float[]>(packet.Payload);
					string vectorHash = SHAHash(vector);
					EmbeddingNode? node = Nodes.FirstOrDefault(n => SHAHash(n.Vector) == vectorHash);

					if (node == null)
					{
						ctx.Log(ID, 2, $"Failed to associate vector. Node with hash '{vectorHash}' not found.");

						Packet response = new()
						{
							TargetID = packet.SourceID,
							SourceID = ID,
							Type = "memory:embedding-engine->associate-response",
							PayloadType = "error",
							Payload = MessagePackSerializer.Serialize("Node not found"),
							Success = false
						};
						ctx.Send(response);
					}
					else
					{
						node.Associations[vectorHash] = (vector, float.Parse(packet.Data["weight"]));
						float[][] vectors = node.Associations.Values.Select(vec => vec.Item1).ToArray();
						float[] average = AverageVectors(vectors);
						node.Vector = NormalizeVector(average);

						ctx.Log(ID, 3, $"Associated vector with hash '{vectorHash}' to node. Updated node vector.");

						Packet response = new()
						{
							TargetID = packet.SourceID,
							SourceID = ID,
							Type = "memory:embedding-engine->associate-response",
							PayloadType = "node",
							Payload = MessagePackSerializer.Serialize(node),
							Success = true
						};
						ctx.Send(response);
					}
				}
				else if (packet.Type == "memory:embedding-engine->find-similar")
				{
					ctx.Log(ID, 3, "Received 'find-similar' request. (Implementation pending)");
				}
				else
				{
					ctx.Log(ID, 2, $"Unknown packet type '{packet.Type}' received.");
				}
			};

			List<Task> tasks = [];

			while (ctx.ShouldNotExit())
			{
				CheckForInput(this, main, ref tasks);
			}

			Task.WaitAll(tasks.ToArray());
			ctx.Log(ID, 3, "Exiting main loop of EmbeddingEngineModule.");
		}

		private float[] AverageVectors(float[][] vectors)
		{
			float[] average = new float[EmbeddingSize];
			for (int i = 0; i < EmbeddingSize; i++)
			{
				float sum = vectors.Sum(v => v[i]);
				average[i] = sum / vectors.Length;
			}
			return average;
		}

		private float[] NormalizeVector(float[] vector)
		{
			float length = MathF.Sqrt(vector.Sum(v => v * v));
			for (int i = 0; i < vector.Length; i++)
			{
				vector[i] /= length;
			}
			return vector;
		}

		private float CosineSimilarity(float[] vectorA, float[] vectorB)
		{
			float dotProduct = 0;
			float lengthA = 0;
			float lengthB = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				dotProduct += vectorA[i] * vectorB[i];
				lengthA += vectorA[i] * vectorA[i];
				lengthB += vectorB[i] * vectorB[i];
			}
			return dotProduct / (MathF.Sqrt(lengthA) * MathF.Sqrt(lengthB));
		}
	}
}
