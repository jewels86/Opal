using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Opal
{
	public static class Core
	{
		private static int _nextID = -1;
		private static object _nextIDLock = new();

		public static ConcurrentDictionary<int, IModule> RegisteredModules { get; } = new();

		public static Action<string, int, string> Log { get; set; } = StandardLog;

		public static int Register(IModule module)
		{
			int id;
			lock (_nextIDLock) { id = _nextID++; }
			RegisteredModules[id] = module;
			return id;
		}

		public static int[] GetModulesByName(string name)
		{
			return [.. RegisteredModules
				.Where(x => x.Value.Name == name)
				.Select(x => x.Key)];
		}

		public static int[] GetModulesByType<T>()
		{
			return [.. RegisteredModules
				.Where(x => x.Value.GetType() == typeof(T))
				.Select(x => x.Key)];
		}

		public static bool Send(Packet packet)
		{
			if (RegisteredModules.TryGetValue(packet.Target, out var module))
			{
				module.Receive(packet);
				return true;
			}
			return false;
		}

		public static void StandardLog(string name, int level, string message)
		{
			Console.WriteLine($"[{name}] [{level}] {message}");
		}
	}
}
