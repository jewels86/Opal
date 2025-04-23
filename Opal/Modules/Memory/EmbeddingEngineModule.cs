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
		public Dictionary<float[], float> Associations { get; set; } = new();
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
			ctx.Add(this);
		}

		public void Main(Context ctx) 
		{
			Action<int> main = (i) =>
			{
				Span<byte> bytes = new();
				lock (InputLocks[i]) { Inputs[i].Read(bytes); }
				Packet packet = MessagePackSerializer.Deserialize<Packet>(bytes.ToArray());

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
					Packet response = new()
					{
						TargetID = packet.SourceID,
						SourceID = ID,
						Type = "memory:embedding-engine->create-response",
						PayloadType = "node",
						Payload = MessagePackSerializer.Serialize(node)
					};
				}
				else if (packet.Type == "memory:embedding-engine->associate")
				{
					float[] vector = MessagePackSerializer.Deserialize<float[]>(packet.Payload);
					EmbeddingNode? node = Nodes.FirstOrDefault(n => n.Vector.SequenceEqual(vector));
					if (node == null)
					{
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
						node.Associations.Add(vector, float.Parse(packet.Data["weight"]));
						node.Vector = AverageVectors(new[] { node.Vector }.Concat(node.Associations.Keys).ToArray());
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
			};

			List<Task> tasks = [];

			while (ctx.ShouldNotExit()) 
			{
				CheckForInput(this, main, ref tasks);
			}

			Task.WaitAll(tasks.ToArray());
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
	}
}
