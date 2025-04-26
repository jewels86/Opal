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

		public static Packet Create(string targetID, string sourceID, string type, string payloadType, object? payload, bool? success = null, Dictionary<string, string>? data = null)
		{
			return new Packet
			{
				TargetID = targetID,
				SourceID = sourceID,
				Type = type,
				PayloadType = payloadType,
				Payload = payload,
				Success = success,
				Data = data ?? new Dictionary<string, string>()
			};
		}
		public Packet() { }
	}
}
