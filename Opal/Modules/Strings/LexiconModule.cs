using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Strings
{
	public class LexiconModule : IModule
	{
		public string ID => "strings:lexicon";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public List<string> AwaitedResponseTypes { get; } = new()
		{
			"memory:embedding-engine->create-response",
			"memory:embedding-engine->add-metadata-response"
		};
		public ConcurrentQueue<Packet> AwaitedResponses { get; } = new();
		public ConcurrentDictionary<string, int> WordToEmbedding { get; } = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			Action<Packet> func = packet =>
			{
				if (packet == null) return;

				if (packet.Type == "strings:lexicon->add-word" && TypeIs(packet.PayloadType, "string"))
				{
					string word = (string)packet.Payload!;
					ctx.Log(ID, 3, $"Adding word to lexicon: {word}");

					if (WordToEmbedding.TryGetValue(word, out int existingId) && existingId >= 0)
					{
						ctx.Log(ID, 3, $"Word '{word}' already exists with embedding ID {existingId}.");
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->add-word-response",
							Payload = existingId,
							PayloadType = "int",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = true
						});
						return;
					}

					WordToEmbedding[word] = -1;

					Output.Enqueue(new Packet()
					{
						Type = "memory:embedding-engine->create",
						TargetID = "memory:embedding-engine",
						SourceID = ID,
						Payload = null,
						PayloadType = "null"
					});

					if (TryWaitForInput(4000, out Packet? createResponse, AwaitedResponses, p => p.Type == "memory:embedding-engine->create-response"))
					{
						if (createResponse != null && TypeIs(createResponse.PayloadType, "int"))
						{
							int newId = (int)createResponse.Payload!;
							WordToEmbedding[word] = newId;

							Output.Enqueue(new Packet()
							{
								Type = "memory:embedding-engine->add-metadata",
								TargetID = "memory:embedding-engine",
								SourceID = ID,
								Payload = (newId, "word", word),
								PayloadType = "(int, string, string)"
							});

							Output.Enqueue(new Packet()
							{
								Type = "strings:lexicon->add-word-response",
								Payload = newId,
								PayloadType = "int",
								SourceID = ID,
								TargetID = packet.SourceID,
								Success = true
							});
						}
						else
						{
							ctx.Log(ID, 3, $"Invalid response when creating embedding for '{word}'.");
							WordToEmbedding.TryRemove(word, out _);
							Output.Enqueue(new Packet()
							{
								Type = "strings:lexicon->add-word-response",
								Payload = null,
								PayloadType = "null",
								SourceID = ID,
								TargetID = packet.SourceID,
								Success = false
							});
						}
					}
					else
					{
						ctx.Log(ID, 3, $"Timeout waiting for embedding creation for '{word}'.");
						WordToEmbedding.TryRemove(word, out _);
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->add-word-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}
				else if (packet.Type == "strings:lexicon->get-id" && TypeIs(packet.PayloadType, "string"))
				{
					string word = (string)packet.Payload!;
					if (WordToEmbedding.TryGetValue(word, out int id) && id >= 0)
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-id-response",
							Payload = id,
							PayloadType = "int",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = true
						});
					}
					else
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-id-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}
				else if (packet.Type == "strings:lexicon->get-word" && TypeIs(packet.PayloadType, "int"))
				{
					int id = (int)packet.Payload!;
					string? word = WordToEmbedding.FirstOrDefault(x => x.Value == id).Key;
					if (word != null)
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-word-response",
							Payload = word,
							PayloadType = "string",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = true
						});
					}
					else
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-word-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}
				else if (packet.Type == "memory:embedding-engine->add-metadata-response")
				{
					ctx.Log(ID, 3, $"Received metadata response: {packet.Payload}");
				}
				else
				{
					ctx.Log(ID, 3, $"Unhandled packet type: {packet.Type}");
				}
			};

			while (ctx.ShouldNotExit())
			{
				if (Input.TryDequeue(out Packet? packet))
				{
					if (packet == null) continue;
					if (AwaitedResponseTypes.Contains(packet.Type))
					{
						AwaitedResponses.Enqueue(packet);
						ctx.Log(ID, 3, $"Queued awaited response: {packet.Type} (ID {packet.PacketID})");
						continue;
					}
					Task.Run(() => func(packet));
				}
			}
		}
	}
}
