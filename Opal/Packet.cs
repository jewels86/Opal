using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public struct Packet(int sender, int target)
	{
		public int ID { get; set; } = -1;
		public Dictionary<string, object?> Payload { get; set; } = [];
		public int Sender { get; set; } = sender;
		public int Target { get; set; } = target;

		public void Deconstruct(out Dictionary<string, object?> payload, out int sender)
		{
			payload = Payload;
			sender = Sender;
		}
	}
}
