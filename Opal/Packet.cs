using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.Marshalling;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{

	public class Packet
	{
		public required string TargetID { get; set; }
		public required string SourceID { get; set; }
		public required string Type { get; set; }
		public required string PayloadType { get; set; }
		public required object? Payload { get; set; }
		public bool? Success { get; set; } = null;
		public Dictionary<string, string> Data { get; set; } = new();
		public int PacketID { get; set; } = 0;
	}
}
