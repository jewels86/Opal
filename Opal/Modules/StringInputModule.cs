using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Opal.Modules
{
	public class StringInputModule : IModule
	{
		public string ID => "string-input";
		public bool IsInput => true; 
		public Stream Input { get; set; }

		public 

		public void Initialize(Context ctx) { }
		public void Receive(Signal sig) { }
		public void Step(Context ctx)
		{

		}
	}
}
