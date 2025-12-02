using System.Diagnostics;
using Opal;
using Opal.NNs;

namespace Testing;

public class Tests
{
    public static void SequenceMemoryTest()
    {
        Console.WriteLine("Testing LSTM sequence memory (remembering first input)...");
        
        // Task: output the first value seen in the sequence
        var sequencesRaw = new[]
        {
            new[] { new[] { 0.5f }, new[] { 0.1f }, new[] { 0.2f } },
            new[] { new[] { -0.3f }, new[] { 0.4f }, new[] { 0.1f } },
            new[] { new[] { 0.8f }, new[] { -0.2f }, new[] { 0.3f } },
            new[] { new[] { -0.6f }, new[] { 0.2f }, new[] { -0.1f } }
        };
        
        var targetsRaw = new[]
        {
            new[] { 0.5f },
            new[] { -0.3f },
            new[] { 0.8f },
            new[] { -0.6f }
        };

        var sequences = sequencesRaw.Select(Operations.Stack).ToArray();
        var targets = targetsRaw.Select(x => Operations.Stack(x)).ToArray();

        var network = new BatchedVectorLstmNetwork(
            1, 8, 1, 8,
            LossFunctions.MeanSquaredError);

        double initialLoss = network.EvaluateLossSequences([sequences], targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        Stopwatch sw = Stopwatch.StartNew();
        
        network.TrainSequences([sequences], targets, 100, 0.01f);
        
        sw.Stop();
        double finalLoss = network.EvaluateLossSequences([sequences], targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}% ({sw.ElapsedMilliseconds}ms)");
        
        // Console.WriteLine("\nPredictions:");
        // for (int i = 0; i < sequencesR.Length; i++)
        // {
        //     var prediction = network.ForwardSequence(sequences[i]);
        //     var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F1")));
        //     Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F4} (expected {targetsRaw[i][0]:F1})");
        // }
    }
}