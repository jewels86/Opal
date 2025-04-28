using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using static Opal.Utilities.ModuleUtilities;
using MessagePack.Resolvers;
using System.Numerics;

namespace Opal.Modules.Strings
{
	public class NextWordGenerationModule : IModule
	{
		public string ID => "strings:next-word-generation";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public ConcurrentDictionary<string, ConcurrentQueue<Packet>> Responses { get; } = new();
		public List<string> ResponseTypes { get; } = new()
						{
							"memory:embedding-engine->get-id-response",
							"memory:embedding-engine->find-by=metadata-tag-response",	
							"memory:embedding-engine->find-similar-response",
						};
		public bool CanContinue { get; set; } = true;


		public void Initialize(Context ctx)
		{
			ctx.Add(this);
			Responses["memory:embedding-engine->get-id-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-similar-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-by=metadata-tag-response"] = new ConcurrentQueue<Packet>();
		}

		public void Main(Context ctx)
		{
			List<Task> tasks = new();
			Action<Packet> main = (packet) =>
			{
				if (ResponseTypes.Contains(packet.Type))
				{
					Responses[packet.Type].Enqueue(packet);
					ctx.Log(ID, 3, $"Response received: {packet.Type}, requeued");
					return;
				}
				
				if (packet.Type == "strings:next-word-generation->generate" && packet.Payload is (string[], int))
				{
					(string[] tokens, int maxTokens) = ((string[], int))packet.Payload!;
					ctx.Log(ID, 3, $"Generating next word for tokens: {string.Join(", ", tokens)} (with max {maxTokens}");
					ctx.Log(ID, 3, $"Fetching embeddings...");

					Dictionary<string, int> ids = new();
					Dictionary<string, float[]> idToVector = new();


					foreach (string token in tokens)
					{
						
					}
					
				}
			};

			while (ctx.ShouldNotExit())
			{
				CheckForInput(this, main, ref tasks, ctx.DeltaTime);
			}
		}
	}
}
