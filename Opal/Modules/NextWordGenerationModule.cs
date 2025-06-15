using Opal.Utilities.ANNs;
using Opal.Utilities.ANNs.Recurrent;

namespace Opal.Modules;

public class NextWordGenerationModule : IModule
{
    public EmbeddingsModule<string> Embeddings { get; set; }
    public LstmNetwork Lstm { get; private set; } = new("next-word-generation-lstm");

    public int ID { get; private set; }
    public string Name { get; private set; }
    
    public NextWordGenerationModule(string? name = null, EmbeddingsModule<string>? embeddings = null, int hiddenSize = 128,
        int batchSize = 1)
    {
        Name = name ?? "next-word-generation";
        ID = Core.Register(this);
        Embeddings = embeddings ?? new(32, 128, 64, 0.1, "next-word-generation-embeddings");
        int n = Embeddings.N;
        Lstm.AddLayer(new LstmLayer(n, hiddenSize, batchSize));
        Lstm.AddLayer(new LstmLayer(hiddenSize, n, batchSize));
        Lstm.AddLayer(new LstmLayer(n, n, batchSize));
    }
    
    
}