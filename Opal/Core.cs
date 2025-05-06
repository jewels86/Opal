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
		public static int LogLevel { get; set; } = 2;

		public static int Register(IModule module)
		{
			int id;
			lock (_nextIDLock) { id = _nextID++; }
			RegisteredModules[id] = module;
			return id;
		}

		#region Getters
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
		#endregion

		#region Logging
		public static void StandardLog(string name, int level, string message)
		{
			if (level > LogLevel) return;
			Console.WriteLine($"[{name}] [{level}] {message}");
		}
		#endregion

		public static void Initialize()
		{
			foreach (var module in RegisteredModules)
			{
				module.Value.Initialize();
			}
		}
	}
}
