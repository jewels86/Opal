using System.Diagnostics;
using Jewels.Opal;
using Jewels.Opal.NNs;

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

        var sequences = Operations.New(Operations.Stack(sequencesRaw));
        var targets = Operations.New(Operations.Stack(targetsRaw));

        var network = new BatchedVectorLstmNetwork(
            1, 8, 1, 8,
            LossFunctions.MeanSquaredError);

        float initialLoss = network.EvaluateLossFinal(sequences, targets, LossFunctions.MeanSquaredError);
        Console.WriteLine($"Initial loss: {initialLoss}");
        Stopwatch sw = Stopwatch.StartNew();
        
        network.TrainFinal(sequences, targets, LossFunctions.MeanSquaredError, 1000, 0.01f);
        
        sw.Stop();
        float finalLoss = network.EvaluateLossFinal(sequences, targets, LossFunctions.MeanSquaredError);
        Console.WriteLine($"Final loss: {finalLoss} ({sw.ElapsedMilliseconds}ms)");
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