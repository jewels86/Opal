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

		private bool _wait = false;
		private readonly object _waitLock = new object();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
			Responses["memory:embedding-engine->get-id-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-similar-response"] = new ConcurrentQueue<Packet>();
			Responses["memory:embedding-engine->find-by-metadata-tag-response"] = new ConcurrentQueue<Packet>();
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

				if (packet.Type == "strings:next-word-generation->generate" && packet.Payload is (string[], int))
				{
					(string[] tokens, int maxTokens) = ((string[], int))packet.Payload!;
					ctx.Log(ID, 3, $"Generating next word for tokens: {string.Join(", ", tokens)} (with max {maxTokens})");
					ctx.Log(ID, 3, $"Fetching embeddings...");

					Dictionary<string, int> ids = new();
					Dictionary<int, (int, double)[]> similars = [];
					ConcurrentQueue<Packet> inQueue; 

					foreach (string token in tokens)
					{
						if (!ids.TryGetValue(token, out int id))
						{
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
									ids[token] = id;
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
									similars[ids[token]] = ((List<(int, double)>)p.Payload!).OrderBy(s => s.Item2).Reverse().ToArray();
									ctx.Log(ID, 3, $"Similar tokens found for ID {id}: {string.Join(", ", similars[ids[token]].Select(x => $"{x.Item1} ({x.Item2})"))}.");
								}
								else
								{
									ctx.Log(ID, 3, $"Failed to retrieve ID for token '{token}'.");
									return;
								}
							}
						}
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
								ctx.Log(ID, 3, $"Processing packet: {packet.Type} (payload type {packet.PayloadType})");
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
							ctx.Log(ID, 3, $"Response received: {inputPacket.Type}, requeued");
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
