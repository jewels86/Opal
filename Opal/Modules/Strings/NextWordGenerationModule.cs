using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using static Opal.Utilities.ModuleUtilities;
using MessagePack.Resolvers;
using System.Numerics;
using Opal.Modules.Memory;

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
				"memory:embedding-engine->find-by-metadata-tag-response",
				"memory:embedding-engine->find-similar-response",
				"memory:embedding-engine->get-by-id-response",
				"memory:embedding-engine->similarity-response"
			};

		private bool _wait = false;
		private readonly object _waitLock = new object();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
			Responses["memory:embedding-engine->get-id-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-similar-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-by-metadata-tag-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->get-by-id-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->similarity-response"] = new ConcurrentQueue<Packet>();
		}

		public void Main(Context ctx)
		{
			Queue<Packet> resultQueue = new();

			Action<Packet> processPacket = (packet) =>
			{
				if (ResponseTypes.Contains(packet.Type))
				{
					Responses[packet.Type].Enqueue(packet);
					ctx.Log(ID, 3, $"Response received: {packet.Type}, requeued");
					return;
				}

				if (packet.Type == "strings:next-word-generation->generate" && packet.Payload is string[])
				{
					string[] tokens = (string[])packet.Payload!;
					if (tokens.Length == 0)
					{
						ctx.Log(ID, 3, "No tokens provided for next word generation.");
						Output.Enqueue(new Packet()
						{
							Type = "strings:next-word-generation->generate-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = "strings:next-word-generation"
						});
						return;
					}
					ctx.Log(ID, 3, $"Generating next word for tokens: {string.Join(", ", tokens)}");
					ctx.Log(ID, 3, $"Fetching embeddings...");

					int id;
					string token = tokens[^1];
					List<(int, double)> similars = [];
					ConcurrentQueue<Packet> inQueue;

					Output.Enqueue(new Packet()
					{
						Type = "memory:embedding-engine->get-id",
						Payload = token,
						PayloadType = "string",
						SourceID = ID,
						TargetID = "memory:embedding-engine"
					});

					if (TryWaitForInput(4000, out Packet? response, Responses["memory:embedding-engine->get-id-response"], p => p.PayloadType == "int"))
					{
						if (response != null && response.Payload is int retrievedId)
						{
							id = retrievedId;
							ctx.Log(ID, 3, $"Token '{token}' retrieved with ID {id}.");
							ctx.Log(ID, 3, $"Finding similar tokens for ID {id}...");

							Output.Enqueue(new Packet()
							{
								Type = "memory:embedding-engine->find-similar",
								Payload = id,
								PayloadType = "int",
								SourceID = ID,
								TargetID = "memory:embedding-engine"
							});
							inQueue = Responses["memory:embedding-engine->find-similar-response"];
							List<Packet> similar = WaitForExpectedResponses(1, ref inQueue);
							if (similar.Count == 0 || similar[0].Payload is not List<(int, double)>)
							{
								ctx.Log(ID, 3, $"Failed to retrieve similar tokens for ID {id}.");
								return;
							}
							Packet p = similar[0];
							similars.AddRange((List<(int, double)>)p.Payload!);
							ctx.Log(ID, 3, $"Similar tokens found for ID {id}: {string.Join(", ", similars.Select(x => $"{x.Item1} ({x.Item2})"))}.");
						}
						else
						{
							ctx.Log(ID, 3, $"Failed to retrieve ID for token '{token}'.");
							return;
						}
					}

					ctx.Log(ID, 3, $"Finding similar sentences...");
					List<EmbeddingNode> similarSentences = [];
					List<(int, double)> nextWords = [];
					inQueue = Responses["memory:embedding-engine->get-by-id-response"];
					List<(int, double)> toRemove = [];

					foreach (var (similarID, s) in similars)
					{
						Output.Enqueue(new Packet()
						{
							Type = "memory:embedding-engine->get-by-id",
							Payload = similarID,
							PayloadType = "int",
							SourceID = ID,
							TargetID = "memory:embedding-engine"
						});
						ctx.Log(ID, 3, $"Getting ID of {similarID}...");
						inQueue = Responses["memory:embedding-engine->get-by-id-response"];
						Packet res = WaitForExpectedResponses(1, ref inQueue)[0];
						ctx.Log(ID, 3, $"Received response for ID {similarID}.");
						if (res.Payload is EmbeddingNode node)
						{
							if (node.Metadata.ContainsKey("sentence")) 
							{
								toRemove.Add((similarID, s));
								similarSentences.Add(node);
								ctx.Log(ID, 3, $"Found similar sentence: {node.Metadata["sentence"]}");

								int[] words = node.Metadata["sentence"].Split(", ").Select(int.Parse).ToArray();
								if (words.Length == 0)
								{
									ctx.Log(ID, 3, $"No words found in sentence metadata for ID {similarID}.");
									continue;
								}
								List<(int, double)> wordSimilarities = [];
								foreach (var word in words)
								{
									Output.Enqueue(new Packet()
									{
										Type = "memory:embedding-engine->similarity",
										Payload = (similarID, word),
										PayloadType = "(int, int)",
										SourceID = ID,
										TargetID = "memory:embedding-engine"
									});
									inQueue = Responses["memory:embedding-engine->similarity-response"];
									Packet similarityResponse = WaitForExpectedResponses(1, ref inQueue)[0];
									if (similarityResponse.Payload is not double)
									{
										ctx.Log(ID, 3, $"Failed to retrieve similarity for ID {similarID} and word {word}.");
										continue;
									}
									double similarity = (double)similarityResponse.Payload!;
									wordSimilarities.Add((word, similarity));
								}
								wordSimilarities = wordSimilarities.OrderByDescending(x => x.Item2).ToList();
								ctx.Log(ID, 3, $"Similar words for ID {similarID}: {string.Join(", ", wordSimilarities.Select(x => $"{x.Item1} ({x.Item2})"))}.");
								var nextWord = wordSimilarities[0].Item1;
								Output.Enqueue(new Packet()
								{
									Type = "memory:embedding-engine->similarity",
									Payload = (similarID, nextWord),
									PayloadType = "(int, int)",
									SourceID = ID,
									TargetID = "memory:embedding-engine"
								});
								inQueue = Responses["memory:embedding-engine->similarity-response"];
								Packet nextWordResponse = WaitForExpectedResponses(1, ref inQueue)[0];
								if (nextWordResponse.Payload is double nextWordSimilarity)
								{
									ctx.Log(ID, 3, $"Next word for ID {similarID}: {nextWord} with similarity {nextWordSimilarity}.");
									nextWords.Add((nextWord, nextWordSimilarity));
								}
								else
								{
									ctx.Log(ID, 3, $"Failed to retrieve next word for ID {similarID}.");
								}
							}
							else
							{
								ctx.Log(ID, 3, $"No sentence metadata found for ID {similarID}.");
							}
						}
					}
					foreach (var (similarID, s) in toRemove)
					{
						similars.Remove((similarID, s));
					}

					List<(int, double)> overlap = similars.Intersect(nextWords).ToList();
					Random random = new Random();
					var weightedNextWords = nextWords
						.Select(x => (x.Item1, WeightedScore: x.Item2 * (1 + random.NextDouble() * 0.1)))
						.OrderByDescending(x => x.WeightedScore)
						.ToList();
					ctx.Log(ID, 3, $"Weighted next words: {string.Join(", ", weightedNextWords.Select(x => $"{x.Item1} ({x.WeightedScore})"))}.");

					if (weightedNextWords.Count > 0)
					{
						var selectedWord = weightedNextWords[0].Item1;
						ctx.Log(ID, 3, $"Selected next word with randomness: {selectedWord} (weighted score: {weightedNextWords[0].WeightedScore})");

						Output.Enqueue(new Packet()
						{
							Type = "strings:next-word-generation->generate-response",
							Payload = selectedWord,
							PayloadType = "int",
							SourceID = ID,
							TargetID = packet.SourceID
						});
					}
					else
					{
						ctx.Log(ID, 3, "No valid next word found after applying randomness.");
						Output.Enqueue(new Packet()
						{
							Type = "strings:next-word-generation->generate-response",
							Payload = null,
							PayloadType = "null",
							SourceID = ID,
							TargetID = packet.SourceID
						});
					}


				}

				lock (_waitLock) { _wait = false; }
			};

			while (ctx.ShouldNotExit())
			{
				lock (_waitLock)
				{
					if (!_wait)
					{
						if (resultQueue.TryDequeue(out Packet? packet))
						{
							if (packet != null)
							{
								_wait = true;
								ctx.Log(ID, 4, $"Processing packet: {packet.Type} (payload type {packet.PayloadType})");
								Task.Run(() => processPacket(packet));
							}
						}
					}
				}

				if (Input.TryDequeue(out Packet? inputPacket))
				{
					if (inputPacket != null)
					{
						if (ResponseTypes.Contains(inputPacket.Type))
						{
							Responses[inputPacket.Type].Enqueue(inputPacket);
							ctx.Log(ID, 4, $"Response received: {inputPacket.Type}, requeued");
							continue;
						}
						resultQueue.Enqueue(inputPacket);
					}
				}
				else
				{
					Task.Delay(ctx.DeltaTime).Wait();
				}
			}
		}
	}
}
