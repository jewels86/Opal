using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	[MessagePackObject]
	public class Packet
	{
		[Key(0)]
		public required string TargetID { get; set; }
		[Key(1)]
		public required string SenderID { get; set; }
		[Key(2)]
		public required string Type { get; set; }
		[Key(3)]
		public required string PayloadType { get; set; }
		[Key(4)]
		public required byte[] Payload { get; set; }
	}
}
