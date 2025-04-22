using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public class Context
	{
		private LogDelegate _log = Logging.StandardLog;

		public Dictionary<string, IModule> SyncModules { get; } = new();
		public Dictionary<string, IAsyncModule> AsyncModules { get; } = new();

		public void Send(Signal sig)
		{
			if (SyncModules.TryGetValue(sig.To, out var module))
			{
				module.Receive(sig);
			}
			else if (AsyncModules.TryGetValue(sig.To, out var asyncModule))
			{
				asyncModule.ReceiveAsync(sig);
			}
			else
			{
				_log.Invoke("ctx", 2, $"Module not found, {sig.To} (from {sig.From} with type {sig.Type})");
			}
		}
	}
}
