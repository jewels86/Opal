using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using MessagePack;
using Opal.Utilities;

namespace Opal.Modules.Memory
{
	public class SemanticMemoryModule : IModule
	{
		public string ID => "memory:semantic-memory";
		public List<MemoryStream> Inputs { get; } = [new()];
		public List<object> InputLocks { get; } = [new()];
		public List<bool> Available { get; } = [true];

		public SemanticNetwork<float[]> Network { get; } = new();

		public void Initialize(Context ctx) {}

		public void Main(Context ctx)
		{
			while (ctx.ShouldNotExit())
			{
				if (Inputs[0].CanRead)
				{
					Span<byte> bytes = new();
					lock (InputLocks[0])
					{
						Inputs[0].Read(bytes);
					}
					Packet packet = MessagePackSerializer.Deserialize<Packet>(bytes.ToArray());

					if (packet is null) continue;
					if (packet.Type == "memory:semantic-memory->lookup")
					{
						SemanticNetworkNode<float[]>? node = Network.Nodes[MessagePackSerializer.Deserialize<float[]>(packet.Payload)];
						if (node is not null)
						{
							SemanticNetworkNodeData<float[]> data = node.Export();
							Packet response = new()
							{
								Type = "memory:semantic-memory->lookup-response",
								TargetID = packet.SourceID,
								SourceID = ID,
								PayloadType = "semantic-network-node",
								Payload = MessagePackSerializer.Serialize(data),
								Success = true
							};
							ctx.Send(response);
						}
					}
					if (packet.Type == "memory:semantic-memory->connect")
					{
						float[][] ids = MessagePackSerializer.Deserialize<float[][]>(packet.Payload);
						float[] idA = ids[0];
						float[] idB = ids[1];

						float weight = ids[2][0];
						int connection = Network.Connect(idA, idB, weight);
						Packet response = new()
						{
							Type = "memory:semantic-memory->connect-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							PayloadType = "semantic-network-connection",
							Payload = MessagePackSerializer.Serialize(Network.Nodes[idA].Connections[connection].Export()),
							Success = true
						};
						ctx.Send(response);
					}
					if (packet.Type == "memory:semantic-memory->disconnect")
					{
						float[][] ids = MessagePackSerializer.Deserialize<float[][]>(packet.Payload);
						float[] idA = ids[0];
						float[] idB = ids[1];
						bool disconnect = Network.Disconnect(idA, idB);
						Packet response = new()
						{
							Type = "memory:semantic-memory->disconnect-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							PayloadType = "null",
							Payload = [],
							Success = disconnect
						};
						ctx.Send(response);
					}
					if (packet.Type == "memory:semantic-memory->create")
					{
						float[] id = MessagePackSerializer.Deserialize<float[]>(packet.Payload);
						SemanticNetworkNode<float[]> node = Network.GetOrCreateNode(id);
						Packet response = new()
						{
							Type = "memory:semantic-memory->create-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							PayloadType = "semantic-network-node",
							Payload = MessagePackSerializer.Serialize(node.Export()),
							Success = true
						};
						ctx.Send(response);
					}
				}
			}
		}
	}
}
