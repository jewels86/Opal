using MessagePack;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Opal;
using static Opal.Utilities.ModuleUtilities;
using Opal.Modules.Memory;

namespace Testing
{
	public class EmbeddingEngineTestModule : IModule
	{
		public string ID => "test:embedding-engine";
		public List<MemoryStream> Inputs { get; } = new() { new MemoryStream() };
		public List<object> InputLocks { get; } = new() { new object() };
		public List<bool> Available { get; } = new() { true };

		public string Type { get; } = "test";

		private Context _context;
		private float[] _vector1;
		private float[] _vector2;

		public void Initialize(Context ctx)
		{
			_context = ctx;
			ctx.Log(ID, 3, "Initializing EmbeddingEngineTestModule.");

			// Generate two initial random vectors
			_vector1 = GenerateRandomVector(128);
			_vector2 = GenerateRandomVector(128);

			ctx.Log(ID, 3, $"Generated initial vectors: Vector1Hash = {SHAHash(_vector1)}, Vector2Hash = {SHAHash(_vector2)}.");

			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			ctx.Log(ID, 3, "Starting main loop of EmbeddingEngineTestModule.");

			// Test: Associate the first vector
			SendAssociatePacket(ctx, _vector1);

			// Test: Associate the second vector
			SendAssociatePacket(ctx, _vector2);

			// Wait for responses
			ProcessResponses(ctx);

			ctx.Log(ID, 3, "Exiting main loop of EmbeddingEngineTestModule.");
		}

		private void SendAssociatePacket(Context ctx, float[] vector)
		{
			ctx.Log(ID, 3, $"Sending 'associate' packet to EmbeddingEngineModule with vector hash '{SHAHash(vector)}'.");

			Packet associatePacket = new()
			{
				TargetID = "memory:embedding-engine",
				SourceID = ID,
				Type = "memory:embedding-engine->associate",
				PayloadType = "vector",
				Payload = MessagePackSerializer.Serialize(vector),
				Data = new Dictionary<string, string>
					{
						{ "weight", "0.5" }
					}
			};

			ctx.Send(associatePacket);
		}

		private void ProcessResponses(Context ctx)
		{
			ctx.Log(ID, 3, "Processing responses from EmbeddingEngineModule.");

			Action<int> process = (i) =>
			{
				lock (InputLocks[i])
				{
					if (Inputs[i].Length > Inputs[i].Position)
					{
						byte[] buffer = new byte[Inputs[i].Length - Inputs[i].Position];
						Inputs[i].Read(buffer, 0, buffer.Length);

						Packet response = MessagePackSerializer.Deserialize<Packet>(buffer);
						ctx.Log(ID, 3, $"Received response: Type = {response.Type}, Success = {response.Success}");

						if (response.PayloadType == "node")
						{
							EmbeddingNode node = MessagePackSerializer.Deserialize<EmbeddingNode>(response.Payload);
							ctx.Log(ID, 3, $"Node received: VectorHash = {SHAHash(node.Vector)}");
						}
						else if (response.PayloadType == "error")
						{
							string error = MessagePackSerializer.Deserialize<string>(response.Payload);
							ctx.Log(ID, 2, $"Error received: {error}");
						}
					}
				}
			};

			List<Task> tasks = new();
			CheckForInput(this, process, ref tasks);
			Task.WaitAll(tasks.ToArray());
		}

		private float[] GenerateRandomVector(int size)
		{
			Random random = new();
			float[] vector = new float[size];
			for (int i = 0; i < size; i++)
			{
				vector[i] = (float)random.NextDouble();
			}
			return vector;
		}
	}
}
