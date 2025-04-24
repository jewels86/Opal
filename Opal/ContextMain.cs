using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public partial class Context
	{
		public void Start()
		{
			List<Task> tasks = [];
			lock (_syncModulesLock)
			{
				foreach (var module in _syncModules.Values)
				{
					tasks.Add(Task.Run(() => module.Main(this)));
				}
			}
			lock (_asyncModulesLock)
			{
				foreach (var asyncModule in _asyncModules.Values)
				{
					tasks.Add(Task.Run(() => asyncModule.MainAsync(this)));
				}
			}
			Task.WaitAll(tasks.ToArray());
		}
	}
}
