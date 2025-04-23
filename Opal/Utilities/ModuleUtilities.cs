using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
	}
}
