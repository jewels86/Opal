using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Strings
{
	public class SentenceLexiconModule : IModule
	{
		public string ID => "strings:sentence-lexicon";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public List<string> AwaitedResponseTypes { get; } = new()
						{
							"memory:embedding-engine->create-response",
							"memory:embedding-engine->add-metadata-response"
						};
		public ConcurrentQueue<Packet> AwaitedResponses { get; } = new();

		public ConcurrentDictionary<string[], int> SentenceToEmbedding { get; } = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			Action<Packet> func = packet =>
			{
				if (packet == null) return;
				if (packet.Type == "strings:sentence-lexicon->add-sentence" && TypeIs(packet.PayloadType, "string[]"))
				{
					string[] sentence = (string[])packet.Payload!;
					ctx.Log(ID, 3, $"Adding sentence to lexicon: {sentence}");

					if (!SentenceToEmbedding.TryAdd(sentence, -1))
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:sentence-lexicon->add-sentence-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
						return;
					}

					Output.Enqueue(new Packet()
					{
						Type = "memory:embedding-engine->create",
						TargetID = "memory:embedding-engine",
						SourceID = ID,
						Payload = null,
						PayloadType = "null"
					});

					if (TryWaitForInput(4000, out Packet? packet2, AwaitedResponses, p => p.Type == "memory:embedding-engine->create-response"))
					{
						if (packet2 != null && TypeIs(packet2.PayloadType, "int"))
						{
							int id = (int)packet2.Payload!;
							SentenceToEmbedding[sentence] = id;

							Output.Enqueue(new Packet()
							{
								Type = "memory:embedding-engine->add-metadata",
								TargetID = "memory:embedding-engine",
								SourceID = ID,
								Payload = (id, "sentence", sentence),
								PayloadType = "(int, string, string[])"
							});

							Output.Enqueue(new Packet()
							{
								Type = "strings:sentence-lexicon->add-sentence-response",
								Payload = id,
								PayloadType = "int",
								SourceID = ID,
								TargetID = packet!.SourceID,
								Success = true
							});
							return;
						}

						Output.Enqueue(new Packet()
						{
							Type = "strings:sentence-lexicon->add-sentence-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet!.SourceID,
							Success = false
						});
					}
				}
				else if (packet.Type == "strings:sentence-lexicon->get-id" && TypeIs(packet.PayloadType, "string[]"))
				{
					string[] sentence = (string[])packet.Payload!;
					if (SentenceToEmbedding.TryGetValue(sentence, out int id))
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:sentence-lexicon->get-id-response",
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
							Type = "strings:sentence-lexicon->get-id-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}
				else if (packet.Type == "strings:sentence-lexicon->get-sentence" && TypeIs(packet.PayloadType, "int"))
				{
					int id = (int)packet.Payload!;
					string[]? sentence = SentenceToEmbedding.FirstOrDefault(x => x.Value == id).Key;
					if (sentence != null)
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:sentence-lexicon->get-sentence-response",
							Payload = sentence,
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
							Type = "strings:sentence-lexicon->get-sentence-response",
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
					ctx.Log(ID, 3, $"Invalid packet type: {packet.Type}");
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
						ctx.Log(ID, 3, $"Re-queued packet: {packet.Type}");
						continue;
					}
					Task.Run(() => func(packet));
				}
			}
		}
	}
}
