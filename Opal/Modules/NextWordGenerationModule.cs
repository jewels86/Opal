using Opal.Utilities;
using Opal.Utilities.ANNs.Lstm;
using static Opal.Utilities.Logging.LogLevel;
using static Opal.Utilities.Logging.AddedLogLevel;
using static Opal.Utilities.Logging;
namespace Opal.Modules;

public class NextWordGenerationModule<T> : IModule where T : notnull
{ 
    public EmbeddingsModule<T> Embeddings { get; }
    public LstmNetwork Lstm { get; } = new("next-word-generation-lstm");

    public string Name { get; }

    public LogLevel Baseline { get; set; } = Info;
    public bool LoggingEnabled { get; set; } = true;

    public NextWordGenerationModule(string? name = null, EmbeddingsModule<T>? embeddings = null, 
        int hiddenLayers = 1, int hiddenSize = 128, int batchSize = 1)
    {
        Name = name ?? "next-word-generation";
        Embeddings = embeddings ?? new(32, 128, 64, 0.1, "next-word-generation-embeddings");

        int n = Embeddings.EmbeddingSize;
        Lstm.AddLayer(new LstmLayer(n, hiddenSize, batchSize, $"lstm-input-layer ({n}, {hiddenSize}) from {Name}"));
        for (int i = 0; i < hiddenLayers; i++)
            Lstm.AddLayer(new LstmLayer(hiddenSize, hiddenSize, batchSize, $"lstm-hidden-layer {i + 1} ({hiddenSize}, {hiddenSize}) from {Name}"));
        Lstm.AddLayer(new LstmLayer(hiddenSize, n, batchSize, $"lstm-output-layer ({hiddenSize}, {n}) from {Name}"));
    }
    
    public List<double> Train(T[] input, T[] target, int epochs = 100, double learningRate = 0.01)
    {
        if (LoggingEnabled) Log(Name, Baseline.Add(LowBaseline), 
            $"Training LSTM with input \'{string.Join(" ", input)}\', target \'{string.Join(" ", target)}\', epochs {epochs}, learning rate {learningRate}");
        
        if (input.Length == 0 || target.Length == 0)
        {
            if (LoggingEnabled) Log(Name, Baseline.Add(HighBaseline), "Input or target is empty.");
            throw new ArgumentException("Input and target cannot be empty.");
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
            if (LoggingEnabled) Log(Name, Baseline.Add(HighBaseline), $"Input words not found in embeddings: {string.Join(", ", missing)}");
            throw new ArgumentException("Some input words not found in embeddings.");
        }
        if (targetEmbeddings.Any(e => e == null))
        {
            var missing = target.Zip(targetEmbeddings, (w, e) => (w, e))
                .Where(x => x.e == null)
                .Select(x => x.w);
            if (LoggingEnabled) Log(Name, Baseline.Add(HighBaseline), $"Target words not found in embeddings: {string.Join(", ", missing)}");
            throw new ArgumentException("Some target words not found in embeddings.");
        }

        List<List<double[]>> inputSeqs = [inputEmbeddings
            .Where(e => e is not null)
            .Cast<Embedding<T>>()
            .Select(e => e?.Vector)
            .ToList()!];
        List<List<double[]>> targetSeqs = [targetEmbeddings
            .Where(e => e is not null)
            .Cast<Embedding<T>>()
            .Select(e => e?.Vector)
            .ToList()!];

        List<double> result = Lstm.Train(inputSeqs, targetSeqs, epochs, learningRate);
        if (LoggingEnabled) Log(Name, Baseline, $"Training complete for input \'{string.Join(" ", input)}\'.");
        return result;
    }
    
    public T GenerateNext(T[] input)
    {
        if (input.Length == 0)
        {
            Log(Name, Baseline.Add(HighBaseline), "Input is empty.");
            throw new ArgumentException("Input cannot be empty.");
        }

        var inputEmbeddings = input
            .Select(word => Embeddings.GetEmbedding(word))
            .ToList();

        if (inputEmbeddings.Any(e => e == null))
        {
            var missing = input.Zip(inputEmbeddings, (w, e) => (w, e))
                .Where(x => x.e == null)
                .Select(x => x.w);
            var missingEnumerable = missing as T[] ?? missing.ToArray();
            Log(Name, Baseline.Add(HighBaseline), $"Input words not found in embeddings: {string.Join(", ", missingEnumerable)}");
            throw new ArgumentException("Some input words not found in embeddings: " + string.Join(", ", missingEnumerable));
        }

        var inputSeq = inputEmbeddings.Select(e => e!.Vector).ToList();
        var outputSeq = Lstm.PredictSequence(inputSeq);

        if (outputSeq.Count == 0)
        {
            Log(Name, Baseline.Add(HighBaseline), "LSTM output sequence is empty.");
            throw new InvalidOperationException("LSTM output sequence is empty.");
        }

        var lastOutput = outputSeq.Last();
        var similars = Embeddings.FindSimilar(EmbeddingsModule<T>.PlaceholderEmbedding(lastOutput)).ToList();

        if (similars.Count == 0)
        {
            Log(Name, Baseline.Add(HighBaseline), "No similar words found for LSTM output.");
            throw new InvalidOperationException("No similar words found for LSTM output.");
        }

        var topCandidate = similars.First().Item1.Data;
        Log(Name, Baseline, "Top candidate: " + topCandidate);

        if (similars.Count > 1)
        {
            double diff = similars[0].Item2 - similars[1].Item2;
            Log(Name, Baseline, "Second closest candidate difference: " + diff);
        }
        else
        {
            Log(Name, Baseline, "No second candidate available.");
        }

        return topCandidate;
    }
}