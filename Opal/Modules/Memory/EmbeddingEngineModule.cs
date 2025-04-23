using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Memory
{
	public class EmbeddingEngineModule : IModule
	{
		public string ID => "memory:embedding-engine";
		public List<MemoryStream> Inputs { get; } = [new()];
		public List<object> InputLocks { get; } = [new()];
		public List<bool> Available { get; } = [true];

		public void Initialize(Context ctx) 
		{
			ctx.Add(this);
		}

		public void Main(Context ctx) 
		{
			Action<int> main = (i) =>
			{
				Span<byte> bytes = new();
				lock (InputLocks[i]) { Inputs[i].Read(bytes); }
				Packet packet = MessagePackSerializer.Deserialize<Packet>(bytes.ToArray());
				
				if (packet.Type == "memory:embedding-engine->")
			};

			List<Task> tasks = [];

			while (ctx.ShouldNotExit())
			{
				CheckForInput(this, main, ref tasks);
			}

			Task.WaitAll(tasks.ToArray());
		}
	}
}
