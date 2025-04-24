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
			IEnumerable<IInteractable> modules = _syncModules.Values.Cast<IInteractable>()
					.Concat(_asyncModules.Values.Cast<IInteractable>());
			Dictionary<string, IInteractable> moduleDict = new();

			foreach (var m in modules)
			{
				moduleDict.Add(m.ID, m);
			}

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

			while (tasks.All(tasks => tasks.IsCompleted) == false)
			{
				foreach (var module in modules)
				{
					if (module.Output.TryDequeue(out Packet? maybePacket))
					{
						Packet packet = maybePacket!;
						lock (_packetIDCountLock) { packet.PacketID = _packetIDCount; _packetIDCount++; }
						if (moduleDict.TryGetValue(packet.TargetID, out IInteractable? target))
						{
							target.Input.Enqueue(packet);
						}
					}
				}
			}
		}
	}
}
