using Opal.Utilities;
using Opal.Utilities.ANNs;
using static Opal.Utilities.Logging.LogLevel;
using static Opal.Utilities.Logging.AddedLogLevel;
using static Opal.Utilities.Logging;
namespace Opal.Modules;

public class NextWordGenerationModule<T> : IModule where T : notnull
{ 
    public EmbeddingsModule<T> Embeddings { get; }
    public INetwork<double[][],double[][]> Network { get; }

    public string Name { get; }

    public LogLevel Baseline { get; set; } = Info;
    public bool LoggingEnabled { get; set; } = true;

    public NextWordGenerationModule(EmbeddingsModule<T> embeddings, INetwork<double[][], double[][]> rnn, string? name = null)
    {
        Name = name ?? "next-word-generation";
        Embeddings = embeddings;
        Network = rnn;
    }
    
    public void Train(T[] input, T[] target, int epochs = 100, double learningRate = 0.01)
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

        double[][][] inputSeqs = [
            inputEmbeddings
                .Where(e => e is not null)
                .Cast<Embedding<T>>()
                .Select(e => e.Vector)
                .ToArray()
        ];
        double[][][] targetSeqs = [
            targetEmbeddings
                .Where(e => e is not null)
                .Cast<Embedding<T>>()
                .Select(e => e.Vector)
                .ToArray()
        ];

        Network.Train(inputSeqs, targetSeqs, epochs, learningRate);
        if (LoggingEnabled) Log(Name, Baseline, $"Training complete for input \'{string.Join(" ", input)}\'.");
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

        var inputSeq = inputEmbeddings.Select(e => e?.Vector!).ToArray();
        var outputSeq = Network.Forward(inputSeq);

        if (outputSeq.Length == 0)
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