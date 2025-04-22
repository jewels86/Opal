using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public class Signal
	{
		public required string To { get; set; }
		public required string From { get; set; }
		public required string Type { get; set; }
		public required object Content { get; set; }
	}
}
