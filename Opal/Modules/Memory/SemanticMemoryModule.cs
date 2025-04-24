using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using MessagePack;
using Opal.Utilities;

namespace Opal.Modules.Memory
{
	public class SemanticMemoryModule //: IModule
	{
		public string ID => "memory:semantic-memory";
		public List<MemoryStream> Inputs { get; } = [new()];
		public List<object> InputLocks { get; } = [new()];
		public List<bool> Available { get; } = [true];

		public SemanticNetwork<float[]> Network { get; } = new();

		public void Initialize(Context ctx) 
		{
			//ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			
			
		}
	}
}
