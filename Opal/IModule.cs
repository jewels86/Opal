using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Opal.Utilities;

namespace Opal
{
	public interface IModule
	{
		public string Name { get; }
		public Logging.LogLevel Baseline { get; set; }
		public bool LoggingEnabled { get; set; }
	}
}
