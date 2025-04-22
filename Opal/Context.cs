using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MessagePack;

namespace Opal
{
	public class Context
	{
		private LogDelegate _log = Logging.StandardLog;
		private readonly object _logLock = new object();

		private Dictionary<string, IModule> _syncModules = new();
		private Dictionary<string, IAsyncModule> _asyncModules = new();
		private readonly object _syncModulesLock = new object();
		private readonly object _asyncModulesLock = new object();

		private bool _exit { get; set; } = false;
		private readonly object _exitLock = new object();

		public void Add(IInteractable interactable)
		{
			if (interactable is IModule module)
			{
				lock (_syncModulesLock) { _syncModules.Add(module.ID, module); }
			}
			else if (interactable is IAsyncModule asyncModule)
			{
				lock (_asyncModulesLock) { _asyncModules.Add(asyncModule.ID, asyncModule); }
			}
		}
		public void Send(Packet packet)
		{
			lock (_syncModulesLock)
			{
				if (_syncModules.TryGetValue(packet.TargetID, out var module))
				{
					module.Input.Write(MessagePackSerializer.Serialize(packet));
				}
			}
			lock (_asyncModulesLock)
			{
				if (_asyncModules.TryGetValue(packet.TargetID, out var asyncModule))
				{
					asyncModule.Input.Write(MessagePackSerializer.Serialize(packet));
				}
			}
		}

		public void Log(string ID, int level, string content)
		{
			_log.Invoke(ID, level, content);
		}
		public void SetLog(LogDelegate log)
		{
			lock (_logLock) { _log = log; }
		}

		public void Exit()
		{
			lock (_exitLock) { _exit = true; }
		}
		public bool ShouldExit() { return _exit; }
	}
}
