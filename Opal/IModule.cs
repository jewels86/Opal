using System;
using System.Collections.Concurrent;
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
		public ConcurrentQueue<Packet> Input { get; }
		public ConcurrentQueue<Packet> Output { get; }
	}
	public interface IModule : IInteractable
	{
		public void Initialize(Context ctx);
		public void Main(Context ctx);
	}
	public interface IAsyncModule : IInteractable
	{
		public void Initialize(Context ctx);
		public Task MainAsync(Context ctx);
	}
}
