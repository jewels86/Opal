using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
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

		public Dictionary<string, int> WordToID { get; } = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			List<Task> tasks = [];
			Action<Packet> func = (packet) =>
			{
				if (packet.Type == "semantic-interpreter->interpret" && packet.Payload is string[])
				{
					ctx.Log(ID, 3, $"Interpreting tokens: {string.Join(", ", (string[])packet.Payload!)}");
					string[] tokens = (string[])packet.Payload!;

					List<int> tokenIDs = [];
					foreach (var token in tokens)
					{
						if (!WordToID.TryGetValue(token, out int id))
						{
							ctx.Log(ID, 3, $"Token '{token}' not found in lexicon. Failed.");
							Output.Enqueue(new Packet()
							{
								Type = "semantic-interpreter->interpret-response",
								Payload = null,
								PayloadType = "null",
								SourceID = ID,
								TargetID = packet.SourceID,
								Success = false
							});
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
						Output.Enqueue(new Packet()
						{
							Type = "memory:embedding-engine->associate",
							Payload = (tokenIDs[i], associations),
							PayloadType = "(int, Dictionary<int, double>)",
							SourceID = ID,
							TargetID = "memory:embedding-engine"
						});
					}
				}
			};

			while (ctx.ShouldNotExit())
			{
				CheckForInput(this, func, ref tasks);
			}
		}
	}
}
