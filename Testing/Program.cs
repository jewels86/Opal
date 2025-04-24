using Opal;
using Opal.Modules;

namespace Testing
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Context ctx = new();
			var embeddings = new Opal.Modules.Memory.EmbeddingEngineModule();
			var embeddingsTest = new EmbeddingEngineTestModule();
			embeddings.Initialize(ctx);
			embeddingsTest.Initialize(ctx);

			ctx.Start();
		}
	}
}
