using MessagePack;
using System;
using System.Collections.Concurrent;
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
		public int ID { get; set; }
		[Key(1)]
		public float[] Vector { get; set; } = new float[128];
		[Key(2)]
		public Dictionary<int, float> Associations { get; set; } = new();
		[Key(3)]
		public Dictionary<string, float> Metadata { get; set; } = new();
	}

	public class EmbeddingEngineModule : IModule
	{
		public string ID => "memory:embedding-engine";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();


		public int EmbeddingSize { get; } = 128;
		public List<EmbeddingNode> Nodes { get; } = new();

		private Random _random = new();
		private int _nextNodeID = 1; 

		public void Initialize(Context ctx)
		{
			ctx.Log(ID, 3, "Initializing EmbeddingEngineModule.");
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			ctx.Log(ID, 3, "Starting main loop of EmbeddingEngineModule.");

			Action<Packet> main = (packet) =>
			{
				if (packet.Type == "memory:embedding-engine->create")
				{
					float[] vector = new float[EmbeddingSize];
					for (int i = 0; i < EmbeddingSize; i++)
					{
						vector[i] = (float)_random.NextDouble();
					}
					int newID = _nextNodeID++;
					Nodes.Add(new EmbeddingNode { ID = newID, Vector = vector });
					ctx.Log(ID, 3, $"Created new embedding node with ID: {newID}");
					Output.Enqueue(new Packet
					{
						Type = "memory:embedding-engine->create-response",
						TargetID = packet.SourceID,
						SourceID = ID,
						Payload = newID,
						PayloadType = "int",
						PacketID = -packet.PacketID,
					});
				}
				else if (packet.Type == "memory:embedding-engine->associate")
				{
					if (TypeIs(packet.PayloadType, "(int, Dictionary<int, float>)"))
					{
						var payload = (ValueTuple<int, Dictionary<int, float>>)packet.Payload!;
						int nodeID = payload.Item1;
						Dictionary<int, float> associations = payload.Item2;
						EmbeddingNode? node = Nodes.FirstOrDefault(n => n.ID == nodeID);
						if (node == null)
						{
							ctx.Log(ID, 2, $"Node with ID {nodeID} not found.");
							return;
						}
						foreach (var kvp in associations) 
						{
							node.Associations[kvp.Key] = kvp.Value;
						}
						ctx.Log(ID, 3, $"Associated {associations.Count} vectors with node {nodeID}-{SHAHash(node.Vector)} (hashed SHA256)");
						ctx.Log(ID, 3, $"Adjusting vector...");
						float[] averageVector = AverageVectors(Nodes.Where(n => associations.ContainsKey(n.ID)).Select(n => n.Vector).ToArray());
						averageVector = AverageVectors([node.Vector, averageVector]);
						float[] normalized = NormalizeVector(averageVector);
						node.Vector = normalized;
						ctx.Log(ID, 3, $"New vector for node {nodeID}: {SHAHash(normalized)}");
						Output.Enqueue(new Packet
						{
							Type = "embedding:embedding-engine->associate-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = true,
							PayloadType = "bool",
							PacketID = -packet.PacketID,
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for associate: {packet.PayloadType} (should be (int, Dictionary<int, float>)");
					}
				}
				else
				{
					ctx.Log(ID, 2, $"Unknown packet type: {packet.Type}");
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
