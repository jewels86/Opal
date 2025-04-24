using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MessagePack;

namespace Opal
{
	public partial class Context
	{
		private LogDelegate _log = Logging.StandardLog;
		private readonly object _logLock = new object();
		private int _loglevel = 3;

		private int _packetIDCount = 1;
		private readonly object _packetIDCountLock = new object();

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

		public void Log(string ID, int level, string content)
		{
			if (level > _loglevel) return;
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
