using Opal.Utilities;
using Opal.Modules;
using System.Numerics;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			SemanticInterpreterModule semanticInterpreter = new();
			semanticInterpreter.AddWord("hello");
			semanticInterpreter.AddWord("world");

			semanticInterpreter.Interpret(["hello", "world"]);
			Console.WriteLine(semanticInterpreter.GetSimilarity("hello", "world"));
		}
	}
}
