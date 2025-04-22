using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public interface IModule
	{
		public string ID { get; }
		public string IsInput { get; }
		public void Initialize(Context ctx);
		public void Receive(Signal sig);
		public List<Signal> Step(Context ctx);
	}
	public interface IAsyncModule
	{
		public string ID { get; }
		public string IsInput { get; }
		public bool IsFinished { get; }

		public Task InitializeAsync(Context ctx);
		public Task ReceiveAsync(Signal sig);
		public Task StepAsync(Context ctx);
	}
}
