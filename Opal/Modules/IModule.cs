using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Modules
{
	public interface IModule
	{
		public string ID { get; }
		public void Initialize(Context ctx);
		public void Receive(Signal sig);
		List<Signal> Emit();
	}
	public interface IAsyncModule
	{
		public string ID { get; }
		public Task InitializeAsync(Context ctx);
		public Task ReceiveAsync(Signal sig);
		Task<List<Signal>> EmitAsync();
	}
}
