using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Security.Cryptography;
using MessagePack;
using System.Collections.Concurrent;

namespace Opal.Utilities
{
	public static class ModuleUtilities
	{
		public static void CheckForInput(IInteractable self, Action<Packet> func, ref List<Task> tasks, int milliseconds = 400)
		{
			if (self.Input.TryDequeue(out Packet? result))
			{
				if (result != null)
				{
					tasks.Add(Task.Run(() => func(result)));
				}
			}
			else
			{
				Task.Delay(milliseconds).Wait();
			}
		}

		public static string SHAHash(double[] vector)
		{
			string str = Convert.ToBase64String(SHA256.HashData(MessagePackSerializer.Serialize(vector)));
			return str;
		}
		public static bool TypeIs(string type, string target)
		{
			return TypeIs(type, [target]);
		}
		public static bool TypeIs(string type, string[] types)
		{
			foreach (var t in types)
			{
				if (type.ToLower().Replace(" ", "") == t.ToLower().Replace(" ", ""))
				{
					return true;
				}
				
			}
			return false;
		}

		public static bool TryWaitForInput(int milliseconds, out Packet? packet, ConcurrentQueue<Packet> queue, Func<Packet, bool>? selection = null)
		{
			packet = null;
			var stopwatch = System.Diagnostics.Stopwatch.StartNew();

			while (stopwatch.ElapsedMilliseconds < milliseconds)
			{
				if (queue.TryDequeue(out var dequeuedPacket))
				{
					if (selection == null || selection(dequeuedPacket))
					{
						packet = dequeuedPacket;
						return true;
					}
				}

				Task.Delay(10).Wait();
			}

			return false;
		}
		public static List<Packet> WaitForExpectedResponses(int expectedResponses, ConcurrentQueue<Packet> input)
		{
			List<Packet> responses = new();
			int count = 0;
			while (count < expectedResponses)
			{
				if (input.TryDequeue(out Packet? packet))
				{
					if (packet != null)
					{
						responses.Add(packet);
						count++;
					}
				}
				else
				{
					Task.Delay(100).Wait();
				}
			}
			return responses;
		}
	}
}
