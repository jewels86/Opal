using Opal;
using Opal.Mathematics;
using Opal.NNs.Ff;
using static Opal.Autograd.Operations;

namespace Testing;

public static class GpuTest
{
    public static void TestGpuBufferZeroing()
    {
        Console.WriteLine("\n=== Testing GPU Buffer Zeroing ===");
        Console.WriteLine($"GPU Available: {GpuAvailable}");
        
        if (!GpuAvailable)
        {
            Console.WriteLine("GPU not available, skipping test");
            return;
        }

        // Test 1: Simple vector operation
        Console.WriteLine("\nTest 1: Vector operations");
        var v1 = NewVector([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        var v2 = NewVector([4.0, 5.0, 6.0], [0.0, 0.0, 0.0]);
        var result = Add(v1, v2);
        
        Console.WriteLine($"v1 + v2 = [{string.Join(", ", result.Value.ToHost())}]");
        Console.WriteLine($"Expected: [5, 7, 9]");
        
        // Test 2: Simple feedforward network
        Console.WriteLine("\nTest 2: Small network training");
        double[] inputs = [0.5, -0.5];
        double[] targets = [1, -1];

        VectorFfNetwork network = new(
            1, 4, 1, 0,
            ActivationFunctions.IdentityVector, 
            ActivationFunctions.IdentityVector, 
            LossFunctions.MeanSquaredErrorVector);
        
        var inputStorage = inputs.Select(x => NewDefaultVectorStorage([x])).ToArray();
        var targetStorage = targets.Select(x => NewDefaultVectorStorage([x])).ToArray();
        
        Console.WriteLine($"Initial weights (first 3): {string.Join(", ", network.InputLayer.Weights.Value.ToHost().Cast<double>().Take(3))}");
        
        double initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        if (initialLoss < 0)
        {
            Console.WriteLine("❌ ERROR: Loss is negative! Buffer not zeroed properly.");
        }
        else if (double.IsNaN(initialLoss))
        {
            Console.WriteLine("❌ ERROR: Loss is NaN! Garbage data in buffers.");
        }
        else
        {
            Console.WriteLine("✓ Loss is valid");
        }
        
        // Train for a few iterations
        network.Train(inputStorage, targetStorage, 10, 0.01);
        
        double finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Loss after 10 epochs: {finalLoss}");
        
        if (finalLoss < 0 || double.IsNaN(finalLoss))
        {
            Console.WriteLine("❌ ERROR: Loss became invalid during training!");
        }
        else if (finalLoss < initialLoss)
        {
            Console.WriteLine("✓ Loss decreased - training is working!");
        }
        else
        {
            Console.WriteLine("⚠ Warning: Loss did not decrease");
        }
        
        // Dispose to return buffers to pool
        result.Dispose();
        
        // Test 3: Verify buffers are zeroed when reused
        Console.WriteLine("\nTest 3: Buffer reuse after disposal");
        var v3 = NewVector([10.0, 20.0, 30.0], [0.0, 0.0, 0.0]);
        var v4 = NewVector([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        var result2 = Add(v3, v4);
        
        var values = result2.Value.ToHost();
        Console.WriteLine($"v3 + v4 = [{string.Join(", ", values)}]");
        Console.WriteLine($"Expected: [10, 20, 30]");
        
        bool correct = Math.Abs(values[0] - 10.0) < 0.001 && 
                      Math.Abs(values[1] - 20.0) < 0.001 && 
                      Math.Abs(values[2] - 30.0) < 0.001;
        
        if (correct)
        {
            Console.WriteLine("✓ Buffers are properly zeroed on reuse!");
        }
        else
        {
            Console.WriteLine("❌ ERROR: Buffers contain garbage data!");
        }
    }
}

