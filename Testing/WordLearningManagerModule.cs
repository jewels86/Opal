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

		public List<string> SentenceList { get; } = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			Task.Delay(1000).Wait();
			ctx.Log(ID, 3, "WordLearningManagerModule initialized.");
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
				if (TryWaitForInput(4000, out Packet? packet, Input, p => p.Type == "strings:string-parsing->parse-response"))
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
						}
					}
				}
				else
				{
					ctx.Log(ID, 3, $"Failed to parse sentence: {sentence} (no response)");
				}

				Output.Enqueue(new Packet()
				{
					Type = "strings:lexicon->add-sentence",
					Payload = sentence,
					PayloadType = "string",
					SourceID = ID,
					TargetID = "strings:lexicon"
				});
				ctx.Log(ID, 3, $"Adding sentence to lexicon: {sentence}");
			}
		}
	}
}
