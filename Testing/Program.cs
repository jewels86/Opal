using Opal;
using Opal.Modules;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Context ctx = new();
			var embeddings = new Opal.Modules.Memory.EmbeddingEngineModule();
			
			embeddings.Initialize(ctx);
			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->create",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = null,
				PayloadType = "null"
			});

			Task task = Task.Run(ctx.Start);

			Task.Delay(1000).Wait();
			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->create",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = null,
				PayloadType = "null"
			});

			Task.Delay(1000).Wait();
			Dictionary<int, float> associations = new()
			{
				{ 2, 0.5f },
			};
			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->associate",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = (1, associations),
				PayloadType = "(int, Dictionary<int, float>)"
			});
			Task.WaitAll(task);
		}
	}
}
