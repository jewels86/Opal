using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Opal.Utilities;
using static Opal.Utilities.Logging;

namespace Opal
{
	public static class Core
	{
		private static int _nextID = -1;
		private static object _nextIDLock = new();

		public static ConcurrentDictionary<int, IModule> RegisteredModules { get; } = new();

		public static Action<string, int, string> LogFunction { get; set; } = StandardLog;
		public static readonly List<string> LogWhitelist = [];
		public static readonly List<string> LogBlacklist = [];
		
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
		public static void Log(string name, int level, string message)
		{
			LogFunction(name, level, message);
		}

		public static void Log(string name, Logging.LogLevel level, string message)
		{
			LogFunction(name, (int)level, message);
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
