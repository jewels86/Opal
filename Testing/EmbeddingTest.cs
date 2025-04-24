using Opal;
using Opal.Modules;

namespace Testing
{
	internal class EmbeddingTest
	{
		public static void Run(string[] args)
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

			embeddings.Input.Enqueue(new Packet()
			{
				Type = "memory:embedding-engine->add-metadata",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = (1, "word", "fire"),
				PayloadType = "(int, string, string)"
			});

			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->create",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = null,
				PayloadType = "null"
			});
			Task.Delay(1000).Wait(1000);

			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->add-metadata",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = (2, "word", "burn"),
				PayloadType = "(int, string, string)"
			});

			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->find-similar",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = 1,
				PayloadType = "int",
				Data = new Dictionary<string, string> { { "method", "pearson" } }
			});

			Task.Delay(1000).Wait();
			Dictionary<int, double> associations = new()
			{
				{ 2, 0.5 },
			};
			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->associate",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = (1, associations),
				PayloadType = "(int, Dictionary<int, double>)"
			});

			Task.Delay(2000).Wait();
			embeddings.Input.Enqueue(new Packet
			{
				Type = "memory:embedding-engine->find-similar",
				TargetID = embeddings.ID,
				SourceID = "test:source",
				Payload = 1,
				PayloadType = "int",
				Data = new Dictionary<string, string> { { "method", "pearson" } }
			});

			Task.WaitAll(task);
		}
	}
}
