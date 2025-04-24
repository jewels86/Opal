using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Strings
{
	public class StringParsingModule : IModule
	{
		public string ID => "strings:string-parsing";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();
		private static readonly char[] separators = [' ', '.', ',', ';', ':', '!', '?', '-', '_', '(', ')', '[', ']', '{', '}', '\'', '\"', '/', '\\', '|', '\n', '\r', '\t'];

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			while (ctx.ShouldNotExit())
			{
				if (Input.TryDequeue(out Packet? packet))
				{
					if (packet == null) continue;
					if (packet.Type == "strings:string-parsing->parse" && TypeIs(packet.PayloadType, "string"))
					{
						ctx.Log(ID, 3, $"Parsing string: {packet.Payload}");
						string parsedString = ParseString((string)packet.Payload!);
						string[] tokenized = Tokenize(parsedString);
						Output.Enqueue(new Packet()
						{
							Type = "strings:string-parsing->parse-response",
							Payload = tokenized,
							PayloadType = "string[]",
							SourceID = ID,
							TargetID = packet.SourceID,
							Success = true
						});
						ctx.Log(ID, 3, $"Parsed string: {packet.Payload} -> {string.Join(", ", tokenized)}");
					}
					else
					{
						ctx.Log(ID, 3, $"Invalid packet type: {packet.GetType()}");
					}
				}
			}
		}

		private string ParseString(string input)
		{
			return input
				.Replace("\n", "")
				.Replace("\r", "")
				.Replace("\t", "")
				.Replace("\0", "")
				.ToLower();
		}
		private string[] Tokenize(string input)
		{
			return input
				.Split(separators, StringSplitOptions.RemoveEmptyEntries);
		}
	}
}
