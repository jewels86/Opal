using Opal;
using Opal.NNs;

namespace Testing;

public class LstmTests
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

        var sequences = sequencesRaw.Select(x => x.Select(Operations.NewValue).ToArray()).ToArray();
        var targets = targetsRaw.Select(Operations.NewValue).ToArray();

        var network = new VectorLstmNetwork(
            1, 8, 1, 8,
            LossFunctions.MeanSquaredError);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.01f);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}%");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var prediction = network.ForwardSequence(sequences[i]);
            var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F1")));
            Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F4} (expected {targetsRaw[i][0]:F1})");
        }
    }

    public static void CountingTest()
    {
        Console.WriteLine("\nTesting LSTM counting (count 1s in binary sequence)...");
        
        // Task: count how many 1s appear in a binary sequence
        var sequencesRaw = new[]
        {
            new[] { new[] { 0.0 }, new[] { 0.0 }, new[] { 0.0 } },  // 0 ones
            new[] { new[] { 1.0 }, new[] { 0.0 }, new[] { 0.0 } },  // 1 one
            new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 } },  // 1 one
            new[] { new[] { 1.0 }, new[] { 1.0 }, new[] { 0.0 } },  // 2 ones
            new[] { new[] { 1.0 }, new[] { 0.0 }, new[] { 1.0 } },  // 2 ones
            new[] { new[] { 1.0 }, new[] { 1.0 }, new[] { 1.0 } },  // 3 ones
            new[] { new[] { 0.0 }, new[] { 0.0 }, new[] { 1.0 } },  // 1 one
            new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 } }   // 2 ones
        };
        
        var targetsRaw = new[]
        {
            new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 }, new[] { 2.0 },
            new[] { 2.0 }, new[] { 3.0 }, new[] { 1.0 }, new[] { 2.0 }
        };

        // Convert to VectorTensorStorage
        var sequences = sequencesRaw.Select(seq => seq.Select(v => Operations.NewCpuVectorStorage(v)).ToArray()).ToArray();
        var targets = targetsRaw.Select(t => Operations.NewCpuVectorStorage(t)).ToArray();

        var network = new VectorLstmNetwork(
            1, 16, 1, 8,
            ActivationFunctions.SigmoidVector,
            ActivationFunctions.TanhVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 3000, 0.05);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var prediction = network.ForwardSequence(sequences[i]);
            var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F0")));
            Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F2} (expected {targetsRaw[i][0]:F0})");
        }
    }

    public static void SequenceSumTest()
    {
        Console.WriteLine("\nTesting LSTM sequence sum...");
        
        // Task: sum all values in the sequence
        var sequencesRaw = new[]
        {
            new[] { new[] { 0.1 }, new[] { 0.2 }, new[] { 0.3 } },     // 0.6
            new[] { new[] { 0.5 }, new[] { 0.5 }, new[] { 0.0 } },     // 1.0
            new[] { new[] { -0.2 }, new[] { 0.3 }, new[] { 0.1 } },    // 0.2
            new[] { new[] { 0.4 }, new[] { -0.1 }, new[] { 0.2 } },    // 0.5
            new[] { new[] { 0.0 }, new[] { 0.0 }, new[] { 0.5 } },     // 0.5
            new[] { new[] { 0.3 }, new[] { 0.3 }, new[] { 0.3 } }      // 0.9
        };
        
        var targetsRaw = sequencesRaw.Select(seq => new[] { seq.Sum(x => x[0]) }).ToArray();

        // Convert to VectorTensorStorage
        var sequences = sequencesRaw.Select(seq => seq.Select(v => Operations.NewCpuVectorStorage(v)).ToArray()).ToArray();
        var targets = targetsRaw.Select(t => Operations.NewCpuVectorStorage(t)).ToArray();

        var network = new VectorLstmNetwork(
            1, 8, 1, 8,
            ActivationFunctions.SigmoidVector,
            ActivationFunctions.TanhVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.05);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var prediction = network.ForwardSequence(sequences[i]);
            var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F1")));
            Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F3} (expected {targetsRaw[i][0]:F1})");
        }
    }

    public static void SequenceClassificationTest()
    {
        Console.WriteLine("\nTesting LSTM sequence classification...");
        
        // Task: classify sequences as "increasing" [1,0] or "decreasing" [0,1]
        var sequencesRaw = new[]
        {
            new[] { new[] { 0.1 }, new[] { 0.3 }, new[] { 0.5 } },  // increasing
            new[] { new[] { 0.2 }, new[] { 0.4 }, new[] { 0.6 } },  // increasing
            new[] { new[] { 0.8 }, new[] { 0.5 }, new[] { 0.2 } },  // decreasing
            new[] { new[] { 0.9 }, new[] { 0.6 }, new[] { 0.3 } },  // decreasing
            new[] { new[] { 0.0 }, new[] { 0.2 }, new[] { 0.4 } },  // increasing
            new[] { new[] { 0.7 }, new[] { 0.4 }, new[] { 0.1 } }   // decreasing
        };
        
        var targetsRaw = new[]
        {
            new[] { 1.0, 0.0 },
            new[] { 1.0, 0.0 },
            new[] { 0.0, 1.0 },
            new[] { 0.0, 1.0 },
            new[] { 1.0, 0.0 },
            new[] { 0.0, 1.0 }
        };

        // Convert to VectorTensorStorage
        var sequences = sequencesRaw.Select(seq => seq.Select(v => Operations.NewCpuVectorStorage(v)).ToArray()).ToArray();
        var targets = targetsRaw.Select(t => Operations.NewCpuVectorStorage(t)).ToArray();

        var network = new VectorLstmNetwork(
            1, 8, 2, 8,
            ActivationFunctions.SigmoidVector,
            ActivationFunctions.TanhVector,
            LossFunctions.CrossEntropy);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.1);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var output = network.ForwardSequence(sequences[i]).ToHost();
            string predicted = output[0] > output[1] ? "increasing" : "decreasing";
            string expected = targetsRaw[i][0] > targetsRaw[i][1] ? "increasing" : "decreasing";
            Console.WriteLine($"  Sequence {i + 1}: {predicted} (confidence: {Math.Max(output[0], output[1]):F3}) - expected {expected}");
        }
    }

    public static void LongTermMemoryTest()
    {
        Console.WriteLine("\nTesting LSTM long-term memory (remember first of 5 values)...");
        
        // Task: remember the first value over a longer sequence (LSTM should excel at this)
        var sequencesRaw = new[]
        {
            new[] { new[] { 0.9 }, new[] { 0.1 }, new[] { 0.2 }, new[] { 0.3 }, new[] { 0.4 } },
            new[] { new[] { -0.8 }, new[] { 0.5 }, new[] { 0.2 }, new[] { 0.1 }, new[] { 0.3 } },
            new[] { new[] { 0.7 }, new[] { -0.3 }, new[] { 0.4 }, new[] { 0.2 }, new[] { 0.1 } },
            new[] { new[] { -0.5 }, new[] { 0.2 }, new[] { -0.1 }, new[] { 0.4 }, new[] { 0.3 } }
        };
        
        var targetsRaw = new[]
        {
            new[] { 0.9 },
            new[] { -0.8 },
            new[] { 0.7 },
            new[] { -0.5 }
        };

        // Convert to VectorTensorStorage
        var sequences = sequencesRaw.Select(seq => seq.Select(v => Operations.NewCpuVectorStorage(v)).ToArray()).ToArray();
        var targets = targetsRaw.Select(t => Operations.NewCpuVectorStorage(t)).ToArray();

        var network = new VectorLstmNetwork(
            1, 12, 1, 8,
            ActivationFunctions.SigmoidVector,
            ActivationFunctions.TanhVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 3000, 0.01);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}%");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var prediction = network.ForwardSequence(sequences[i]);
            var seqString = string.Join(", ", sequencesRaw[i].Select(x => x[0].ToString("F1")));
            Console.WriteLine($"  [{seqString}] → {prediction.ToHost()[0]:F4} (expected {targetsRaw[i][0]:F1})");
        }
    }

    public static void RunAll()
    {
        SequenceMemoryTest();
        CountingTest();
        SequenceSumTest();
        SequenceClassificationTest();
        LongTermMemoryTest();
        Console.WriteLine("\n✓ All LSTM tests completed!");
    }
}