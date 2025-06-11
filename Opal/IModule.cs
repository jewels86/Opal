using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public interface IModule
	{
		public int ID { get; }
		public string Name { get; }
		public void Initialize() { }
	}
}
