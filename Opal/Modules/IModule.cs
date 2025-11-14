using Opal.Utilities;

namespace Opal.Modules
{
	public interface IModule
	{
		public string Name { get; }
		public Logging.LogLevel Baseline { get; set; }
		public bool LoggingEnabled { get; set; }
	}
}
