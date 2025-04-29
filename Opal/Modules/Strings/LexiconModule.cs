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
		public ConcurrentDictionary<int[], int> SentenceToEmbedding { get; } = new(new IntArrayEqualityComparer());

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			Action<Packet> func = packet =>
			{
				if (packet == null) return;

				// Handle adding a word
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

					for (int retryCount = 0; retryCount < 4; retryCount++)
					{
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

								bool metadataResponseReceived = false;
								while (!metadataResponseReceived)
								{
									if (TryWaitForInput(4000, out Packet? metadataResponse, AwaitedResponses, p => p.Type == "memory:embedding-engine->add-metadata-response"))
									{
										if (metadataResponse != null && metadataResponse.Payload != null)
										{
											ctx.Log(ID, 3, $"Word '{word}' added with embedding ID {newId}.");
											metadataResponseReceived = true;
										}
										else
										{
											ctx.Log(ID, 3, $"Failed to add metadata for word '{word}'.");
										}
									}
									else
									{
										ctx.Log(ID, 3, $"Timeout waiting for metadata response for word '{word}'. Retrying...");
									}
								}

								Output.Enqueue(new Packet()
								{
									Type = "strings:lexicon->add-word-response",
									Payload = newId,
									PayloadType = "int",
									SourceID = ID,
									TargetID = packet.SourceID,
									Success = true
								});
								return;
							}
							else
							{
								ctx.Log(ID, 3, $"Invalid response from embedding engine: {createResponse?.PayloadType}");
							}
						}
					}
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

				// Handle adding a sentence
				else if (packet.Type == "strings:lexicon->add-sentence" && TypeIs(packet.PayloadType, "string[]"))
				{ 
					string[] words = (string[])packet.Payload!;
					string wordsString = string.Join(", ", words);
					List<int> wordIds = new();

					ctx.Log(ID, 3, $"Adding sentence to lexicon: {wordsString}");

					foreach (string word in words)
					{
						if (!WordToEmbedding.TryGetValue(word, out int wordId))
						{
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
									wordId = (int)createResponse.Payload!;
									WordToEmbedding[word] = wordId;

									Output.Enqueue(new Packet()
									{
										Type = "memory:embedding-engine->add-metadata",
										TargetID = "memory:embedding-engine",
										SourceID = ID,
										Payload = (wordId, "word", word),
										PayloadType = "(int, string, string)"
									});
								}
								else
								{
									ctx.Log(ID, 3, $"Failed to create embedding for word '{word}'.");
									Output.Enqueue(new Packet()
									{
										Type = "strings:lexicon->add-sentence-response",
										Payload = null,
										PayloadType = "null",
										SourceID = ID,
										TargetID = packet.SourceID,
										Success = false
									});
									return;
								}
							}
						}
						wordIds.Add(wordId);
					}

					int[] sentenceIdArray = wordIds.ToArray();
					if (!SentenceToEmbedding.TryGetValue(sentenceIdArray, out int sentenceId))
					{
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
								sentenceId = (int)createResponse.Payload!;
								SentenceToEmbedding[sentenceIdArray] = sentenceId;

								Output.Enqueue(new Packet()
								{
									Type = "memory:embedding-engine->add-metadata",
									TargetID = "memory:embedding-engine",
									SourceID = ID,
									Payload = (sentenceId, "sentence", string.Join(", ", sentenceIdArray)),
									PayloadType = "(int, string, string)"
								});

								ctx.Log(ID, 3, $"Sentence '{wordsString}' added with embedding ID {sentenceId}.");
							}
							else
							{
								ctx.Log(ID, 3, $"Failed to create embedding for sentence '{wordsString}'.");
								Output.Enqueue(new Packet()
								{
									Type = "strings:lexicon->add-sentence-response",
									Payload = null,
									PayloadType = "null",
									SourceID = ID,
									TargetID = packet.SourceID,
									Success = false
								});
								return;
							}
						}
					}

					Output.Enqueue(new Packet()
					{
						Type = "strings:lexicon->add-sentence-response",
						Payload = sentenceId,
						PayloadType = "int",
						SourceID = ID,
						TargetID = packet.SourceID,
						Success = true
					});
				}

				// Handle retrieving a word by ID
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

				// Handle retrieving a sentence ID
				else if (packet.Type == "strings:lexicon->get-sentence-id" && TypeIs(packet.PayloadType, "string"))
				{
					string sentence = (string)packet.Payload!;
					string[] words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
					int[] wordIds = words.Select(word => WordToEmbedding.TryGetValue(word, out int id) ? id : -1).ToArray();

					if (SentenceToEmbedding.TryGetValue(wordIds, out int sentenceId))
					{
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-sentence-id-response",
							Payload = sentenceId,
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
							Type = "strings:lexicon->get-sentence-id-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}

				// Handle retrieving a sentence by ID
				else if (packet.Type == "strings:lexicon->get-sentence" && TypeIs(packet.PayloadType, "int"))
				{
					int sentenceId = (int)packet.Payload!;
					int[]? wordIds = SentenceToEmbedding.FirstOrDefault(x => x.Value == sentenceId).Key;

					if (wordIds != null)
					{
						string sentence = string.Join(" ", wordIds.Select(id => WordToEmbedding.FirstOrDefault(x => x.Value == id).Key));
						Output.Enqueue(new Packet()
						{
							Type = "strings:lexicon->get-sentence-response",
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
							Type = "strings:lexicon->get-sentence-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = false
						});
					}
				}

				// Handle unhandled packet types
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
						ctx.Log(ID, 4, $"Queued awaited response: {packet.Type} (ID {packet.PacketID})");
						continue;
					}
					Task.Run(() => func(packet));
				}
			}
		}
	}
}
