using Opal.Autograd;
using Opal.Mathematics;
using Opal.NNs.Recurrent;

namespace Testing;

public class RecurrentTests
{
    public static void SequenceMemoryTest()
    {
        Console.WriteLine("Testing RNN sequence memory (remembering first input)...");
        Operations.GpuAvailable = false;
        // Task: output the first value seen in the sequence
        // Sequences: [0.5, 0.1, 0.2] → 0.5, [-0.3, 0.4, 0.1] → -0.3
        var sequences = new[]
        {
            new[] { new[] { 0.5 }, new[] { 0.1 }, new[] { 0.2 } },
            new[] { new[] { -0.3 }, new[] { 0.4 }, new[] { 0.1 } },
            new[] { new[] { 0.8 }, new[] { -0.2 }, new[] { 0.3 } },
            new[] { new[] { -0.6 }, new[] { 0.2 }, new[] { -0.1 } }
        };
            
        var targets = new[]
        {
            new[] { 0.5 }, 
            new[] { -0.3 }, 
            new[] { 0.8 }, 
            new[] { -0.6 }
        };

        var network = new VectorRecurrentNetwork(
            1, 8, 1, 1,
            ActivationFunctions.TanhVector,
            ActivationFunctions.IdentityVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.01);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}%");
        
        Console.WriteLine("\nPredictions:");
        foreach (var seq in sequences)
        {
            double[] prediction = network.ForwardSequence(seq);
            Console.WriteLine($"  [{string.Join(", ", seq.Select(x => x[0].ToString("F1")))}] → {prediction[0]:F4} (expected {seq[0][0]:F1})");
        }
    }

    public static void CountingTest()
    {
        Console.WriteLine("\nTesting RNN counting (count 1s in binary sequence)...");
        
        // Task: count how many 1s appear in a binary sequence
        var sequences = new[]
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
        
        var targets = new[]
        {
            new[] { 0.0 },
            new[] { 1.0 },
            new[] { 1.0 },
            new[] { 2.0 },
            new[] { 2.0 },
            new[] { 3.0 },
            new[] { 1.0 },
            new[] { 2.0 }
        };

        var network = new VectorRecurrentNetwork(
            1, 16, 1, 1,
            ActivationFunctions.TanhVector,
            ActivationFunctions.IdentityVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 3000, 0.05);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            double[] prediction = network.ForwardSequence(sequences[i]);
            Console.WriteLine($"  [{string.Join(", ", sequences[i].Select(x => x[0].ToString("F0")))}] → {prediction[0]:F2} (expected {targets[i][0]:F0})");
        }
    }

    public static void SequenceSumTest()
    {
        Console.WriteLine("\nTesting RNN sequence sum...");
        
        // Task: sum all values in the sequence
        var sequences = new[]
        {
            new[] { new[] { 0.1 }, new[] { 0.2 }, new[] { 0.3 } },     // 0.6
            new[] { new[] { 0.5 }, new[] { 0.5 }, new[] { 0.0 } },     // 1.0
            new[] { new[] { -0.2 }, new[] { 0.3 }, new[] { 0.1 } },    // 0.2
            new[] { new[] { 0.4 }, new[] { -0.1 }, new[] { 0.2 } },    // 0.5
            new[] { new[] { 0.0 }, new[] { 0.0 }, new[] { 0.5 } },     // 0.5
            new[] { new[] { 0.3 }, new[] { 0.3 }, new[] { 0.3 } }      // 0.9
        };
        
        var targets = sequences.Select(seq => new[] { seq.Sum(x => x[0]) }).ToArray();

        var network = new VectorRecurrentNetwork(
            1, 8, 1, 1,
            ActivationFunctions.TanhVector,
            ActivationFunctions.IdentityVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.05);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            double[] prediction = network.ForwardSequence(sequences[i]);
            Console.WriteLine($"  [{string.Join(", ", sequences[i].Select(x => x[0].ToString("F1")))}] → {prediction[0]:F3} (expected {targets[i][0]:F1})");
        }
    }

    public static void VectorSequenceClassificationTest()
    {
        Console.WriteLine("\nTesting Vector RNN sequence classification...");
        
        // Task: classify sequences as "increasing" [1,0] or "decreasing" [0,1]
        var sequences = new[]
        {
            new[] { new[] { 0.1 }, new[] { 0.3 }, new[] { 0.5 } },  // increasing
            new[] { new[] { 0.2 }, new[] { 0.4 }, new[] { 0.6 } },  // increasing
            new[] { new[] { 0.8 }, new[] { 0.5 }, new[] { 0.2 } },  // decreasing
            new[] { new[] { 0.9 }, new[] { 0.6 }, new[] { 0.3 } },  // decreasing
            new[] { new[] { 0.0 }, new[] { 0.2 }, new[] { 0.4 } },  // increasing
            new[] { new[] { 0.7 }, new[] { 0.4 }, new[] { 0.1 } }   // decreasing
        };
        
        var targets = new[]
        {
            new[] { 1.0, 0.0 },
            new[] { 1.0, 0.0 },
            new[] { 0.0, 1.0 },
            new[] { 0.0, 1.0 },
            new[] { 1.0, 0.0 },
            new[] { 0.0, 1.0 }
        };

        var network = new VectorRecurrentNetwork(
            1, 8, 2, 1,
            ActivationFunctions.TanhVector,
            ActivationFunctions.SoftmaxVector,
            LossFunctions.CrossEntropy);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 2000, 0.1);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < sequences.Length; i++)
        {
            var output = network.ForwardSequence(sequences[i]);
            string predicted = output[0] > output[1] ? "increasing" : "decreasing";
            string expected = targets[i][0] > targets[i][1] ? "increasing" : "decreasing";
            Console.WriteLine($"  Sequence {i + 1}: {predicted} (confidence: {Math.Max(output[0], output[1]):F3}) - expected {expected}");
        }
    }

    public static void SequenceEchoTest()
    {
        Console.WriteLine("\nTesting RNN echo (output previous input)...");
        
        // Task: echo the previous value in the sequence
        // For sequence [a, b, c], outputs should be [0, a, b]
        // We'll test on the last output only (should equal the second-to-last input)
        var sequences = new[]
        {
            new[] { new[] { 0.5 }, new[] { 0.8 }, new[] { 0.2 } },   // should output 0.8
            new[] { new[] { 0.3 }, new[] { 0.1 }, new[] { 0.6 } },   // should output 0.1
            new[] { new[] { 0.9 }, new[] { 0.4 }, new[] { 0.7 } },   // should output 0.4
            new[] { new[] { 0.2 }, new[] { 0.6 }, new[] { 0.3 } }    // should output 0.6
        };
        
        var targets = sequences.Select(seq => new[] { seq[seq.Length - 2][0] }).ToArray();

        var network = new VectorRecurrentNetwork(
            1, 12, 1, 1,
            ActivationFunctions.TanhVector,
            ActivationFunctions.IdentityVector,
            LossFunctions.MeanSquaredErrorVector);

        double initialLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.TrainSequences(sequences, targets, 3000, 0.01);
        
        double finalLoss = network.EvaluateLossSequences(sequences, targets);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nPredictions (should output previous value):");
        for (int i = 0; i < sequences.Length; i++)
        {
            double[] prediction = network.ForwardSequence(sequences[i]);
            Console.WriteLine($"  [..., {sequences[i][^2][0]:F1}, {sequences[i][^1][0]:F1}] → {prediction[0]:F3} (expected {targets[i][0]:F1})");
        }
    }

    public static void RunAll()
    {
        SequenceMemoryTest();
        CountingTest();
        SequenceSumTest();
        VectorSequenceClassificationTest();
        SequenceEchoTest();
        Console.WriteLine("\n✓ All RNN tests completed!");
    }
}