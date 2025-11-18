using Opal.Autograd;
using Opal.Mathematics;
using Opal.NNs.Ff;
using static Opal.Autograd.Operations;

namespace Testing;

public class DiagnosticTest
{
    public static void TestLossComputation()
    {
        Console.WriteLine("=== Diagnostic Test: Loss Computation ===\n");
        
        // Simple test: create a vector and compute MSE loss
        double[] predicted = [1.0, 2.0, 3.0];
        double[] actual = [1.5, 2.5, 3.5];
        
        var predictedTensor = NewVector(predicted, Vectors.Zeros(3));
        var actualStorage = NewDefaultVectorStorage(actual);
        
        var lossTensor = LossFunctions.MeanSquaredErrorVector(predictedTensor, actualStorage);
        
        Sync(); // Make sure GPU operations complete
        double lossValue = lossTensor.Value.ToHost();
        
        Console.WriteLine($"Predicted: [{string.Join(", ", predicted)}]");
        Console.WriteLine($"Actual: [{string.Join(", ", actual)}]");
        Console.WriteLine($"Loss: {lossValue}");
        
        // Manual calculation: MSE = mean((predicted - actual)^2)
        double expectedLoss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            expectedLoss += diff * diff;
        }
        expectedLoss /= predicted.Length;
        
        Console.WriteLine($"Expected Loss: {expectedLoss}");
        Console.WriteLine($"Match: {Math.Abs(lossValue - expectedLoss) < 0.001}");
        
        lossTensor.Dispose();
    }
    
    public static void TestNetworkForward()
    {
        Console.WriteLine("\n=== Diagnostic Test: Network Forward Pass ===\n");
        
        VectorFfNetwork network = new(
            1, 2, 1, 0,  // 1 input, 2 hidden, 1 output, 0 hidden layers (just input->output)
            ActivationFunctions.IdentityVector, 
            ActivationFunctions.IdentityVector, 
            LossFunctions.MeanSquaredErrorVector);
        
        // Check initial weights
        Sync();
        double[,] weights = network.InputLayer.Weights.Value.ToHost();
        double[] biases = network.OutputLayer.Biases.Value.ToHost();
        
        Console.WriteLine($"Input Layer Weights shape: {weights.GetLength(0)}x{weights.GetLength(1)}");
        Console.WriteLine("Input Layer Weights:");
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            Console.Write($"  Row {i}: ");
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                Console.Write($"{weights[i, j]:F4} ");
            }
            Console.WriteLine();
        }
        
        Console.WriteLine($"\nOutput Layer Biases: [{string.Join(", ", biases.Select(b => $"{b:F4}"))}]");
        
        // Forward pass
        double[] input = [0.5];
        var inputStorage = NewDefaultVectorStorage(input);
        var inputTensor = NewVector(inputStorage, null, _ => { }, NewDefaultVectorStorage(Vectors.Zeros(1)));
        
        var output = network.Forward(inputTensor);
        Sync();
        double[] outputValues = output.Value.ToHost();
        
        Console.WriteLine($"\nInput: [{string.Join(", ", input)}]");
        Console.WriteLine($"Output: [{string.Join(", ", outputValues.Select(v => $"{v:F4}"))}]");
        
        // Test loss
        double[] target = [1.0];
        var targetStorage = NewDefaultVectorStorage(target);
        
        var lossTensor = LossFunctions.MeanSquaredErrorVector(output, targetStorage);
        Sync();
        double lossValue = lossTensor.Value.ToHost();
        
        Console.WriteLine($"Target: [{string.Join(", ", target)}]");
        Console.WriteLine($"Loss: {lossValue}");
        
        // Backward pass
        Console.WriteLine("\nTesting backward pass...");
        lossTensor.Backward(NewDefaultScalarStorage(1.0));
        Sync();
        
        double[,] weightGrad = network.InputLayer.Weights.Gradient.ToHost();
        Console.WriteLine("Weight gradients:");
        for (int i = 0; i < weightGrad.GetLength(0); i++)
        {
            Console.Write($"  Row {i}: ");
            for (int j = 0; j < weightGrad.GetLength(1); j++)
            {
                Console.Write($"{weightGrad[i, j]:F6} ");
            }
            Console.WriteLine();
        }
        
        output.Dispose();
        lossTensor.Dispose();
    }
    
    public static void TestEvaluateLoss()
    {
        Console.WriteLine("\n=== Diagnostic Test: EvaluateLoss Function ===\n");
        
        VectorFfNetwork network = new(
            1, 2, 1, 0,
            ActivationFunctions.IdentityVector, 
            ActivationFunctions.IdentityVector, 
            LossFunctions.MeanSquaredErrorVector);
        
        double[] inputs = [0.5, -0.5];
        double[] targets = [1.0, -1.0];
        
        Console.WriteLine("Creating input storage...");
        var inputStorage = inputs.Select(x => NewDefaultVectorStorage([x])).ToArray();
        
        Console.WriteLine("Creating target storage...");
        var targetStorage = targets.Select(x => NewDefaultVectorStorage([x])).ToArray();
        
        Console.WriteLine("\nVerifying storage contents immediately after creation:");
        for (int i = 0; i < targetStorage.Length; i++)
        {
            Sync();
            double[] values = targetStorage[i].ToHost();
            Console.WriteLine($"  targetStorage[{i}]: [{string.Join(", ", values)}] (expected [{targets[i]}])");
        }
        
        Console.WriteLine($"\nTesting with {inputStorage.Length} samples");
        Console.WriteLine($"Inputs: [{string.Join(", ", inputs)}]");
        Console.WriteLine($"Targets: [{string.Join(", ", targets)}]");
        
        double loss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Computed loss: {loss}");
        
        Console.WriteLine("\nVerifying storage contents after EvaluateLoss:");
        for (int i = 0; i < targetStorage.Length; i++)
        {
            Sync();
            double[] values = targetStorage[i].ToHost();
            Console.WriteLine($"  targetStorage[{i}]: [{string.Join(", ", values)}] (expected [{targets[i]}])");
        }
        
        // Manually compute the loss for verification
        Console.WriteLine("\nManual verification:");
        for (int i = 0; i < inputs.Length; i++)
        {
            var input = inputStorage[i];
            var target = targetStorage[i];
            
            Sync();
            double[] targetValues = target.ToHost();
            Console.WriteLine($"  Sample {i}: target before forward = [{string.Join(", ", targetValues)}]");
            
            var inputTensor = NewVector(input, null, _ => { }, NewDefaultVectorStorage(Vectors.Zeros(1)));
            var output = network.Forward(inputTensor);
            Sync();
            
            double[] outputValues = output.Value.ToHost();
            targetValues = target.ToHost();
            
            Console.WriteLine($"    input={inputs[i]:F2}, output={outputValues[0]:F4}, target={targetValues[0]:F2}");
            
            var lossTensor = LossFunctions.MeanSquaredErrorVector(output, target);
            Sync();
            double sampleLoss = lossTensor.Value.ToHost();
            Console.WriteLine($"    Loss: {sampleLoss:F6}");
            
            output.Dispose();
            lossTensor.Dispose();
        }
    }

    public static void RunAll()
    {
        TestLossComputation();
        TestNetworkForward();
        TestEvaluateLoss();
    }
}

