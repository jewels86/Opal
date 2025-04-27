using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Strings
{
	public class NextWordGenerationModule : IModule
	{
		public string ID => "strings:next-word-generation";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public ConcurrentDictionary<string, ConcurrentQueue<Packet>> Responses { get; } = new()
		{
			["memory:embedding-engine->get-id-response"] = new ConcurrentQueue<Packet>(),
			["memory:embedding-engine->find-similar-response"] = new ConcurrentQueue<Packet>(),
		};
		public List<string> ResponseTypes { get; } = new()
						{
							"memory:embedding-engine->get-id-response",
						};

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
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
					ctx.Log(ID, 3, $"Generating next word for tokens: {string.Join(", ", tokens)}");
					
					string lastWord = tokens[^1];
					Output.Enqueue(Packet.Create(
						"memory:embedding-engine",
						ID,
						"memory:embedding-engine->find-by-metadata-tag",
						"(string, string)",
						("word", lastWord)
					));
					if (TryWaitForInput(4000, out Packet? embeddingResponse, Responses["memory:embedding-engine->get-id-response"]))
					{
						if (embeddingResponse == null || embeddingResponse.Success == false)
						{
							ctx.Log(ID, 3, $"Failed to retrieve embedding for '{lastWord}'.");
							Output.Enqueue(Packet.Create(
								packet.SourceID,
								ID,
								"strings:next-word-generation->generate-response",
								"null",
								null,
								success: false
							));
							return;
						}
						int[] wordIDs = (int[])embeddingResponse.Payload!;
						int wordID = wordIDs[0];
						ctx.Log(ID, 3, $"Embedding for '{lastWord}' retrieved with ID {wordID}.");
						Output.Enqueue(Packet.Create(
							"memory:embedding-engine",
							ID,
							"memory:embedding-engine->find-similar",
							"int",
							wordID
						));
						if (TryWaitForInput(4000, out Packet? similarResponse, Responses["memory:embedding-engine->find-similar-response"]))
						{
							if (similarResponse == null || similarResponse.Success == false)
							{
								ctx.Log(ID, 3, $"Failed to retrieve similar words for ID {wordID}.");
								Output.Enqueue(Packet.Create(
									packet.SourceID,
									ID,
									"strings:next-word-generation->generate-response",
									"null",
									null,
									success: false
								));
								return;
							}
							List<string> similarWords = (List<string>)similarResponse.Payload!;
							ctx.Log(ID, 3, $"Similar words for '{lastWord}': {string.Join(", ", similarWords)} (IDs)");

						}
						else
						{
							ctx.Log(ID, 3, $"Failed to retrieve similar words for ID {wordID}.");
						}
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
