using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public delegate void LogDelegate(string ID, int level, string content);

	public static class Logging
	{
		public static void StandardLog(string ID, int level, string content)
		{
			Console.WriteLine($"[{ID}] [{level}] -> {content}");
		}
	}
}
