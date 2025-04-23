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
		public void Send(Packet packet, int stream = -1)
		{
			lock (_syncModulesLock)
			{
				if (_syncModules.TryGetValue(packet.TargetID, out var module))
				{
					if (stream != -1)
					{
						lock (module.InputLocks[stream])
						{
							module.Inputs[stream].Write(MessagePackSerializer.Serialize(packet));
						}
					}
					else
					{
						int i = 0;
						foreach (bool availability in module.Available)
						{
							if (availability)
							{
								lock (module.InputLocks[i])
								{
									module.Inputs[i].Write(MessagePackSerializer.Serialize(packet));
								}
								return;
							}
							i++;
						}

						lock (module.InputLocks[0])
						{
							module.Inputs[0].Write(MessagePackSerializer.Serialize(packet));
						}
					}
				}
			}
			lock (_asyncModulesLock)
			{
				if (_asyncModules.TryGetValue(packet.TargetID, out var asyncModule))
				{
					if (stream != -1)
					{
						asyncModule.Inputs[stream].Write(MessagePackSerializer.Serialize(packet));
					}
					else
					{
						// Fallback to input 0 for async modules  
						asyncModule.Inputs[0].Write(MessagePackSerializer.Serialize(packet));
					}
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
		public bool ShouldNotExit() { return !_exit; }
	}
}
