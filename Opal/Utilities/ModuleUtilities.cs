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
		public static void CheckForInput(IInteractable self, Action<int> func, ref List<Task> tasks)
		{
			for (int i = 0; i < self.Inputs.Count; i++)
			{
				if (self.Inputs[i].CanRead)
				{
					tasks.Add(Task.Run(() => func(i)));
				}
			}
		}

		public static string SHAHash(float[] vector)
		{
			string str = Convert.ToBase64String(SHA256.HashData(MessagePackSerializer.Serialize(vector)));
			return str;
		}
	}
}
