using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules
{
	public class SemanticInterpreterModule : IModule
	{
		public string ID => "semantic-interpreter";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public ConcurrentQueue<Packet> Responses { get; } = new();

		public ConcurrentDictionary<string, int> WordToID { get; } = new();

		private bool _wait = false;
		private object _waitLock = new object();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			List<Task> tasks = [];
			Action<Packet> func = (packet) =>
			{
				ctx.Log(ID, 3, $"Received packet: {packet.Type} (payload type {packet.PayloadType})");
				if (packet.Type == "semantic-interpreter->interpret" && packet.Payload is string[])
				{
					ctx.Log(ID, 3, $"Interpreting tokens: {string.Join(", ", (string[])packet.Payload!)}");
					string[] tokens = (string[])packet.Payload!;

					List<int> tokenIDs = [];
					foreach (var token in tokens)
					{
						if (!WordToID.TryGetValue(token, out int id))
						{
							ctx.Log(ID, 3, $"Token '{token}' not found in self lexicon. Attempting to retrieve from embeddings...");
							Output.Enqueue(new Packet()
							{
								Type = "memory:embedding-engine->get-id",
								Payload = token,
								PayloadType = "string",
								SourceID = ID,
								TargetID = "memory:embedding-engine"
							});
							if (TryWaitForInput(4000, out Packet? packet2, Input, p => p.Type == "memory:embedding-engine->get-id-response"))
							{
								if (packet2 != null && TypeIs(packet2.PayloadType, "int"))
								{
									id = (int)packet2.Payload!;
									WordToID[token] = id; 
									ctx.Log(ID, 3, $"Token '{token}' retrieved with ID {id}.");
								}
								else
								{
									Output.Enqueue(new Packet()
									{
										Type = "semantic-interpreter->interpret-response",
										Payload = null,
										PayloadType = "null",
										SourceID = ID,
										TargetID = packet.SourceID,
										Success = false
									});
									ctx.Log(ID, 3, $"Failed to retrieve token '{token}'.");
									return;
								}
							}
						}
						tokenIDs.Add(id);
					}
					for (int i = 0; i < tokenIDs.Count; i++)
					{
						Dictionary<int, double> associations = new();
						for (int j = 0; j < tokenIDs.Count; j++)
						{
							if (i == j) continue;
							double strength = 1.0 / Math.Abs(i - j);
							associations[tokenIDs[j]] = strength;
						}
						ctx.Log(ID, 3, $"Associating token ID {tokenIDs[i]} with IDs: {string.Join(", ", associations.Keys)}");
						Output.Enqueue(new Packet()
						{
							Type = "memory:embedding-engine->associate",
							Payload = (tokenIDs[i], associations),
							PayloadType = "(int, Dictionary<int, double>)",
							SourceID = ID,
							TargetID = "memory:embedding-engine"
						});
					}
					var responses = Responses;
					//responses.Clear();
					WaitForExpectedResponses(tokenIDs.Count, ref responses);
					
					Output.Enqueue(new Packet()
					{
						Type = "semantic-interpreter->interpret-response",
						Payload = tokenIDs,
						PayloadType = "int[]",
						SourceID = ID,
						TargetID = packet.SourceID
					});
					ctx.Log(ID, 3, $"Interpreted tokens: {string.Join(", ", tokens)} -> {string.Join(", ", tokenIDs)} (associations created)");
				}
				lock (_waitLock) { _wait = false; }
			};
			Queue<Packet> resultQueue = new();
			while (ctx.ShouldNotExit())
			{
				lock (_waitLock)
				{
					if (!_wait)
					{
						if (resultQueue.TryDequeue(out Packet? res))
						{
							if (res != null)
							{
								_wait = true;
								ctx.Log(ID, 3, $"Processing packet result from queue: {res.Type} (payload type {res.PayloadType})");
								Task.Run(() => func(res));
							}
						}
					}
				}
				if (Input.TryDequeue(out Packet? result))
				{
					if (result != null)
					{
						if (result.Type == "memory:embedding-engine->associate-response")
						{
							Responses.Enqueue(result);
							ctx.Log(ID, 3, $"Received association response: {result.Payload}");
						}
						else { resultQueue.Enqueue(result); }
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
