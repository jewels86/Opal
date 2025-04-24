using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Security.Cryptography;
using MessagePack;

namespace Opal.Utilities
{
	public static class ModuleUtilities
	{
		public static void CheckForInput(IInteractable self, Action<Packet> func, ref List<Task> tasks)
		{
			if (self.Input.TryDequeue(out Packet? result))
			{
				if (result != null)
				{
					tasks.Add(Task.Run(() => func(result)));
				}
			}
		}

		public static string SHAHash(double[] vector)
		{
			string str = Convert.ToBase64String(SHA256.HashData(MessagePackSerializer.Serialize(vector)));
			return str;
		}
		public static bool TypeIs(string type, string target)
		{
			return TypeIs(type, [target]);
		}
		public static bool TypeIs(string type, string[] types)
		{
			foreach (var t in types)
			{
				if (type.ToLower().Replace(" ", "") == t.ToLower().Replace(" ", ""))
				{
					return true;
				}
				
			}
			return false;
		}
	}
}
