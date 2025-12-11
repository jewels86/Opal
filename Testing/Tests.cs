using System.Diagnostics;
using Jewels.Opal;
using Jewels.Opal.NNs;

namespace Testing;

public class Tests
{
    public static void SequenceMemoryTest()
    {
        Console.WriteLine("Testing LSTM sequence memory (remembering first input)...");
        var (numSequences, sequenceLength) = (10, 4);
        
        float[][][] sequencesRaw = new float[numSequences][][];
        for (int i = 0; i < numSequences; i++)
        {
            sequencesRaw[i] = new float[sequenceLength][];
            for (int j = 0; j < sequenceLength; j++) sequencesRaw[i][j] = [Random.Shared.NextSingle() * 2 - 1];
        }
        
        var targetsRaw = sequencesRaw.Select(seq => seq[0]).ToArray();
        var sequences = Operations.New(Operations.Stack(sequencesRaw));
        var targets = Operations.New(Operations.Stack(targetsRaw));
        Console.WriteLine($"Sequences shape: {Operations.ToString(sequences.Value.Shape)}, targets shape: {Operations.ToString(targets.Value.Shape)}");

        var network = new BatchedVectorLstmNetwork(1, 5, 1, 0, LossFunctions.MeanSquaredError);
        float initialLoss = network.EvaluateLossFinal(sequences, targets, LossFunctions.MeanSquaredError);
        Console.WriteLine($"Initial loss: {initialLoss}");
        Stopwatch sw = Stopwatch.StartNew();
        
        network.TrainFinal(sequences, targets, LossFunctions.MeanSquaredError, 4000, 0.01f);
        
        sw.Stop();
        float finalLoss = network.EvaluateLossFinal(sequences, targets, LossFunctions.MeanSquaredError);
        Console.WriteLine($"Final loss: {finalLoss} ({sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}% ({sw.ElapsedMilliseconds}ms)");
        
        // Console.WriteLine("\nPredictions:");
        // for (int i = 0; i < sequencesRaw.Length; i++)
        // {
        //     var prediction = network.ForwardSequenceFinal(sequences[i]);
        //     var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F1")));
        //     Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F4} (expected {targetsRaw[i][0]:F1})");
        // }
    }
    
    
}