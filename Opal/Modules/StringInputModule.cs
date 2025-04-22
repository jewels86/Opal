using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Opal.Modules
{
	public class StringInputModule : IModule
	{
		public MemoryStream Input { get; } = new();
		public string ID { get; } = "string-input";
		public readonly object InputLock = new();

		public void Initialize(Context ctx)
		{
			ctx.Add(this);
		}
		public void Main(Context ctx)
		{
			while (!ctx.ShouldExit())
			{
				if (Input.CanRead)
				{
					Span<byte> bytes = new();
					lock (InputLock)
					{
						Input.Read(bytes);
					}
					string text = Encoding.UTF8.GetString(bytes); //! "Input" is a stream of string here

					ctx
				}
			}
		}
	}
}
