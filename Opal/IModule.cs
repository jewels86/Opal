using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MessagePack;

namespace Opal
{
	public interface IInteractable
	{
		public string ID { get; }
		public List<MemoryStream> Inputs { get; }
		public List<object> InputLocks { get; }
		public List<bool> Available { get; }
	}
	public interface IModule : IInteractable
	{
		public void Initialize(Context ctx);
		public void Main(Context ctx);
	}
	public interface IAsyncModule : IInteractable
	{
		public void Initialize(Context ctx);
		public void MainAsync(Context ctx);
	}
}
