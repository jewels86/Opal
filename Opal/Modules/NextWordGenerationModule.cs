using Opal.Configurations;
using Opal.Utilities;
using Opal.Utilities.ANNs;
using Opal.Utilities.ANNs.Lstm;
using Opal.Utilities.ANNs.Recurrent;

namespace Opal.Modules;

public class NextWordGenerationModule : IModule
{ // TODO: Convert to NextWordGenerationModule<T>
    public EmbeddingsModule<string> Embeddings { get; set; }
    public SemanticInterpreterModule SemanticInterpreter { get; private set; }
    public LstmNetwork Lstm { get; private set; } = new("next-word-generation-lstm");

    public int ID { get; private set; }
    public string Name { get; private set; }
    
    public NextWordGenerationModule(string? name = null, EmbeddingsModule<string>? embeddings = null, 
        SemanticInterpreterModule? semanticInterpreter = null, int hiddenLayers = 1, int hiddenSize = 128, int batchSize = 1)
    {
        Name = name ?? "next-word-generation";
        ID = Core.Register(this);
        Embeddings = embeddings ?? new(32, 128, 64, 0.1, "next-word-generation-embeddings");
        SemanticInterpreter =
            semanticInterpreter ?? SemanticInterpreterConfigurations.GenerateDefaultSemanticInterpreter(Embeddings);
        int n = Embeddings.N;
        Lstm.AddLayer(new LstmLayer(n, hiddenSize, batchSize, $"lstm-input-layer ({n}, {hiddenSize}) from {Name} ({ID})"));
        for (int i = 0; i < hiddenLayers; i++)
            Lstm.AddLayer(new LstmLayer(hiddenSize, hiddenSize, batchSize, $"lstm-hidden-layer {i + 1} ({hiddenSize}, {hiddenSize}) from {Name} ({ID})"));
        Lstm.AddLayer(new LstmLayer(hiddenSize, n, batchSize, $"lstm-output-layer ({hiddenSize}, {n}) from {Name} ({ID})"));
    }
    
    /// <summary>
    /// Trains the LSTM used for next word generation.
    /// </summary>
    /// <param name="input">The input words as an array ("the quick brown")</param>
    /// <param name="target">The target or actual words as an array ("the quick brown fox")</param>
    /// <param name="epochs"></param>
    /// <param name="learningRate"></param>
    /// <returns></returns>
    public List<double> Train(string[] input, string[] target, int epochs = 100, double learningRate = 0.01)
    {
        Core.Log(Name, (int)Logging.LogLevel.HighDebug, $"Training LSTM with input \'{string.Join(" ", input)}\', target \'{string.Join(" ", target)}\', epochs {epochs}, learning rate {learningRate}");
        
        if (input.Length == 0 || target.Length == 0)
        {
            Core.Log(Name, 2, "Input or target is empty.");
            return [];
        }

        var inputEmbeddings = input
            .Select(word => Embeddings.GetEmbedding(word))
            .ToList();
        var targetEmbeddings = target
            .Select(word => Embeddings.GetEmbedding(word))
            .ToList();

        if (inputEmbeddings.Any(e => e == null))
        {
            var missing = input.Zip(inputEmbeddings, (w, e) => (w, e))
                .Where(x => x.e == null)
                .Select(x => x.w);
            Core.Log(Name, 2, $"Input words not found in embeddings: {string.Join(", ", missing)}");
            return [];
        }
        if (targetEmbeddings.Any(e => e == null))
        {
            var missing = target.Zip(targetEmbeddings, (w, e) => (w, e))
                .Where(x => x.e == null)
                .Select(x => x.w);
            Core.Log(Name, 2, $"Target words not found in embeddings: {string.Join(", ", missing)}");
            return [];
        }

        var inputSeqs = new List<List<double[]>> { inputEmbeddings.Select(e => e.Vector).ToList() };
        var targetSeqs = new List<List<double[]>> { targetEmbeddings.Select(e => e.Vector).ToList() };

        return Lstm.Train(inputSeqs, targetSeqs, epochs, learningRate);
    }
    
    public string GenerateNext(string[] input)
    {
        if (input.Length == 0)
        {
            Core.Log(Name, 2, "Input is empty.");
            return string.Empty;
        }

        var inputEmbeddings = input
            .Select(word => Embeddings.GetEmbedding(word))
            .ToList();

        if (inputEmbeddings.Any(e => e == null))
        {
            var missing = input.Zip(inputEmbeddings, (w, e) => (w, e))
                .Where(x => x.e == null)
                .Select(x => x.w);
            Core.Log(Name, 2, $"Input words not found in embeddings: {string.Join(", ", missing)}");
            return string.Empty;
        }

        var inputSeq = inputEmbeddings.Select(e => e.Vector).ToList();
        var outputSeq = Lstm.PredictSequence(inputSeq);

        if (outputSeq.Count == 0)
        {
            Core.Log(Name, 2, "LSTM output sequence is empty.");
            return string.Empty;
        }

        var lastOutput = outputSeq.Last();
        var similars = Embeddings.FindSimilar(EmbeddingsModule<string>.PlaceholderEmbedding(lastOutput)).ToList();

        if (similars.Count == 0)
        {
            Core.Log(Name, 2, "No similar words found for LSTM output.");
            return string.Empty;
        }

        var topCandidate = similars.First().Item1.Data;
        Core.Log(Name, 2, "Top candidate: " + topCandidate);

        if (similars.Count > 1)
        {
            double diff = similars[0].Item2 - similars[1].Item2;
            Core.Log(Name, 3, "Second closest candidate difference: " + diff);
        }
        else
        {
            Core.Log(Name, 3, "No second candidate available.");
        }

        return topCandidate ?? string.Empty;
    }
}