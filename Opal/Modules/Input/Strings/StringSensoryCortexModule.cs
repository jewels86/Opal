using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Modules.Input.Strigs
{
	public class StringSensoryCortexModule : IModule
	{
		public string ID => "string-sensory-cortex";
		public List<MemoryStream> Inputs { get; } = [new()];
		public List<object> InputLocks { get; } = [new()];
		public List<bool> Available { get; } = [true];

		public List<(string, int)> Connections { get; } = [];

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			while (ctx.ShouldNotExit())
			{
				if (Inputs[0].CanRead)
				{
					Span<byte> input = new();
					lock (InputLocks[0])
					{
						Inputs[0].Read(input);
					}

					string text = Encoding.UTF8.GetString(input.ToArray());
					List<string> tokens = [.. text.Split(' ')];

					foreach (var (connection, stream) in Connections)
					{
						ctx.Send(new()
						{
							TargetID = connection,
							SourceID = ID,
							Type = "",
							PayloadType = "list<string>",
							Payload = new byte[0]
						});
					}
				}
			}
		}
	}
}
