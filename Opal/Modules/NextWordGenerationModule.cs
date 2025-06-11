namespace Opal.Modules;

public class NextWordGenerationModule
{
    public EmbeddingsModule<string> Embeddings { get; set; }
    /// <summary>
    /// The context window length for generating the next word (lookback words).
    /// </summary>
    public int N { get; set; }

    public int D => Embeddings.N;
    public int V => Embeddings

}