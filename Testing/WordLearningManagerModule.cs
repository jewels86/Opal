using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Opal;
using static Opal.Utilities.ModuleUtilities;

namespace Testing
{
	internal class WordLearningManagerModule : IModule
	{
		public string ID => "word-learning-manager";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		public ConcurrentQueue<Packet> Responses { get; } = new();

		public List<string> SentenceList { get; } = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			List<Task> tasks = [];
			Action main = () =>
			{
				while (ctx.ShouldNotExit())
				{
					CheckForInput(this, p => Responses.Enqueue(p), ref tasks, ctx.DeltaTime);
				}
			};
			Task mainTask = Task.Run(main);

			Task.Delay(1000).Wait();
			ctx.Log(ID, 3, "WordLearningManagerModule initialized.");
			List<string[]> parsed = [];
			foreach (var sentence in SentenceList)
			{
				Output.Enqueue(new Packet()
				{
					Type = "strings:string-parsing->parse",
					Payload = sentence,
					PayloadType = "string",
					SourceID = ID,
					TargetID = "strings:string-parsing"
				});
				ctx.Log(ID, 3, $"Parsing sentence: {sentence}");
				if (TryWaitForInput(10000, out Packet? packet, Input, p => p.Type == "strings:string-parsing->parse-response"))
				{
					if (packet != null)
					{
						if (TypeIs(packet.PayloadType, "string[]"))
						{
							string[] tokens = (string[])packet.Payload!;
							foreach (var token in tokens)
							{
								Output.Enqueue(new Packet()
								{
									Type = "strings:lexicon->add-word",
									Payload = token,
									PayloadType = "string",
									SourceID = ID,
									TargetID = "strings:lexicon"
								});
								ctx.Log(ID, 3, $"Adding word to lexicon: {token}");
							}
							ctx.Log(ID, 3, $"Parsed sentence: {sentence} -> {string.Join(", ", tokens)}");
							parsed.Add(tokens);
						}
					}
				}
				else
				{
					ctx.Log(ID, 3, $"Failed to parse sentence: {sentence} (no response)");
				}
			}
			Task.WaitAll(tasks.ToArray());
			Task.Delay(4000).Wait();
			foreach (var response in Responses)
			{
				if (response.Type == "strings:lexicon->add-word-response")
				{
					if (response.Payload is not null)
					{
						ctx.Log(ID, 3, $"Word added to lexicon successfully");
					}
					else
					{
						ctx.Log(ID, 3, $"Failed to add word to lexicon");
					}
					ctx.Log(ID, 3, $"Word with index {response.Payload} confirmed");
				}
				else
				{
					ctx.Log(ID, 3, $"Unknown response type: {response.Type}");
				}
			}
			Responses.Clear();
			foreach (var p in parsed)
			{
				Output.Enqueue(new Packet()
				{
					Type = "semantic-interpreter->interpret",
					Payload = p,
					PayloadType = "string[]",
					SourceID = ID,
					TargetID = "semantic-interpreter"
				});
				ctx.Log(ID, 3, $"Interpreting tokens: {string.Join(", ", p)}");
			}
			ctx.Log(ID, 3, $"Waiting for semantic interpreter responses...");
			WaitForExpectedResponses(parsed.Count, Input);
			Console.Write("[!!!!!!!!!!!!!!!] Enter a phrase to continue from: ");
			string? input = Console.ReadLine();
			if (input == null)
			{
				ctx.Log(ID, 3, "No input provided. Exiting.");
				return;
			}
			Console.WriteLine($"Enter the max number of words to generate: ");
			string? maxTokensInput = Console.ReadLine();
			if (maxTokensInput == null || !int.TryParse(maxTokensInput, out int maxTokens))
			{
				ctx.Log(ID, 3, "Invalid input for max tokens. Exiting.");
				return;
			}
			Output.Enqueue(new Packet()
			{
				Type = "strings:string-parsing->parse",
				Payload = input,
				PayloadType = "string",
				SourceID = ID,
				TargetID = "strings:string-parsing"
			});
			ctx.Log(ID, 3, $"Parsing sentence: {input}");
			if (TryWaitForInput(10000, out Packet? parsedPacket, Responses))
			{
				if (parsedPacket == null || parsedPacket.Payload == null)
				{
					ctx.Log(ID, 3, $"Failed to parse sentence: {input}");
				}
				string[] tokens = (string[])parsedPacket!.Payload!;

				Output.Enqueue(new Packet()
				{
					Type = "strings:next-word-generation->generate",
					Payload = (tokens, maxTokens),
					PayloadType = "(string[], int)",
					SourceID = ID,
					TargetID = "strings:next-word-generation"
				});

				Task.Delay(2000).Wait();
				//ctx.Exit();
			}
		}
	}
}
