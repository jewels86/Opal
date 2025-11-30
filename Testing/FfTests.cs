using System.Diagnostics;
using Jewels.Lazulite;
using Opal;
using Opal.NNs;

namespace Testing;

public static class FfTests
{
    private static int _aidx => Operations.DefaultAcceleratorIndex;
    
    public static void OverfittingTest()
    {
        Console.WriteLine($"Training a network to overfit a simple function...");
        float[] inputs = [0.5f, -0.5f];
        float[] targets = [1, -1];

        using VectorFfNetwork network = new(
            1, 1, 1, 1,
            ActivationFunctions.Identity, ActivationFunctions.Identity, 
            LossFunctions.MeanSquaredError);
        
        Value<float[]>[] inputStorage = inputs.Select(x => new VectorValue([x], _aidx)).ToArray<Value<float[]>>();
        Value<float[]>[] targetStorage = targets.Select(x => new VectorValue([x], _aidx)).ToArray<Value<float[]>>();
        Console.WriteLine("Weights sample: " + network.InputLayer.Weights.Value.ToHost()[0, 0]);
        
        OpalContext.GlobalContext.EnsureInitialization();
        Stopwatch sw = Stopwatch.StartNew();
        var initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        sw.Stop();
        
        Console.WriteLine($"Initial loss: {initialLoss} ({sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"Training for 1000 epochs...");
        sw.Restart();
        network.Train(inputStorage, targetStorage, 1000, 0.01f);
        sw.Stop();
        
        Console.WriteLine("Weights sample: " + network.InputLayer.Weights.Value.ToHost()[0, 0]);
        Console.WriteLine($"Evaluating the loss: {network.EvaluateLoss(inputStorage, targetStorage)} ({sw.ElapsedMilliseconds}ms - {sw.ElapsedMilliseconds / 1000f:F2} ms per epoch)");
    }
    
    public static void OverfittingTestBatched()
    {
        Console.WriteLine($"\nTraining a network to overfit a simple function (batched)...");
        float[] inputs = [0.5f, -0.5f];
        float[] targets = [1, -1];

        using BatchedVectorFfNetwork network = new(
            1, 2, 1, 1,
            ActivationFunctions.Identity, ActivationFunctions.Identity, 
            LossFunctions.MeanSquaredError);
    
        Value<float[]>[] inputStorage = inputs.Select(x => new VectorValue([x], _aidx)).ToArray<Value<float[]>>();
        Value<float[]>[] targetStorage = targets.Select(x => new VectorValue([x], _aidx)).ToArray<Value<float[]>>();
    
        Value<float[,]>[] batchedInputs = [Operations.Stack(inputStorage)];
        Value<float[,]>[] batchedTargets = [Operations.Stack(targetStorage)];
        
        OpalContext.GlobalContext.EnsureInitialization();
        Stopwatch sw = Stopwatch.StartNew();
        var initialLoss = network.EvaluateLoss(batchedInputs, batchedTargets);
        sw.Stop();
        
        Console.WriteLine($"Initial loss: {initialLoss} ({sw.ElapsedMilliseconds}ms)");
        Console.WriteLine($"Training for 1000 epochs...");
        sw.Restart();
        network.Train(batchedInputs, batchedTargets, 2000, 0.01f);
        sw.Stop();
        Console.WriteLine($"Evaluating the loss: {network.EvaluateLoss(batchedInputs, batchedTargets)} ({sw.ElapsedMilliseconds}ms)");
        
        var output = network.Forward(batchedInputs[0]).ToHost();
        Console.WriteLine($"Output[0,0] = {output[0, 0]} (target: 1)");
        Console.WriteLine($"Output[1,0] = {output[1, 0]} (target: -1)");
    }
    
    public static void NonlinearFunctionTest()
    {
        Console.WriteLine("\nTraining network on nonlinear function: f(x) = x^2...");
        
        // Generate training data for f(x) = x^2
        var random = new Random(42);
        float[] inputs = new float[20];
        float[] targets = new float[20];
        
        for (int i = 0; i < 20; i++)
        {
            inputs[i] = (float)random.NextDouble() * 4 - 2; // Range [-2, 2]
            targets[i] = inputs[i] * inputs[i];
        }

        using BatchedVectorFfNetwork network = new(
            1, 8, 1, 8,  
            ActivationFunctions.Tanh, ActivationFunctions.Identity, 
            LossFunctions.MeanSquaredError);
        network.DefaultGradClipNorm = 0.1f;
        
        var inputStorage = Operations.Stack(inputs.Select(x => Operations.NewValue([x])).ToArray());
        var targetStorage = Operations.Stack(targets.Select(x => Operations.NewValue([x])).ToArray());
        
        float initialLoss = network.EvaluateLoss([inputStorage], [targetStorage]);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train([inputStorage], [targetStorage], 2000, 0.001f);
        
        float finalLoss = network.EvaluateLoss([inputStorage], [targetStorage]);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}%");
        
        Console.WriteLine("\nTesting predictions:");
        float[] testInputs = [0.0f, 1.0f, -1.5f, 2.0f];
        foreach (var x in testInputs)
        {
            float prediction = network.Forward(Operations.Stack([[x]])).ToHost()[0, 0];
            float expected = x * x;
            Console.WriteLine($"  f({x}) = {prediction:F4} (expected {expected:F4})");
        }
    }

    public static void XorTest()
    {
        Console.WriteLine("\nTraining network on XOR problem...");
        
        float[][] inputs = 
        [
            [0.0f, 0.0f],
            [0.0f, 1.0f],
            [1.0f, 0.0f],
            [1.0f, 1.0f]
        ];
        
        float[][] targets = 
        [
            [0.0f],
            [1.0f],
            [1.0f],
            [0.0f]
        ];

        VectorFfNetwork network = new(
            2, 4, 1, 1, 
            ActivationFunctions.Tanh, 
            ActivationFunctions.Sigmoid, 
            LossFunctions.MeanSquaredError);
        
        var inputStorage = inputs.Select(Operations.NewValue).ToArray();
        var targetStorage = targets.Select(Operations.NewValue).ToArray();
        
        float initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 5000, 0.5f);
        
        float finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nXOR predictions:");
        foreach (var input in inputs)
        {
            float[] output = network.Forward(Operations.NewValue(input)).ToHost();
            Console.WriteLine($"  [{input[0]}, {input[1]}] → {output[0]:F4}");
        }
    }

    public static void IrisClassificationTest()
    {
        Console.WriteLine("\nTraining on simplified Iris dataset (2 classes)...");
        
        // Simplified Iris dataset - just 2 features, 2 classes
        // Features: sepal length, sepal width (normalized)
        float[][] inputs = 
        [
            [0.22f, 0.63f], [0.17f, 0.42f], [0.11f, 0.50f], [0.08f, 0.46f], [0.19f, 0.67f],
            [0.31f, 0.79f], [0.19f, 0.58f], [0.20f, 0.63f], [0.11f, 0.42f], [0.25f, 0.58f],
            [0.69f, 0.42f], [0.56f, 0.54f], [0.61f, 0.42f], [0.53f, 0.33f], [0.56f, 0.50f],
            [0.67f, 0.42f], [0.61f, 0.46f], [0.64f, 0.42f], [0.69f, 0.38f], [0.56f, 0.38f]
        ];
        
        // One-hot encoded classes: [1,0] for setosa, [0,1] for versicolor
        float[][] targets = 
        [
            [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
            [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
        ];

        VectorFfNetwork network = new(
            2, 8, 2, 2,  // 2 inputs, 8 hidden, 2 outputs, 2 hidden layers
            ActivationFunctions.ReLu, 
            ActivationFunctions.Softmax, 
            LossFunctions.CrossEntropy);
        
        var inputStorage = inputs.Select(Operations.NewValue).ToArray();
        var targetStorage = targets.Select(Operations.NewValue).ToArray();
        
        float initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 3000, 0.1f);
        
        float finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        int correct = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            float[] output = network.Forward(Operations.NewValue(inputs[i])).ToHost();
            int predicted = output[0] > output[1] ? 0 : 1;
            int actual = targets[i][0] > targets[i][1] ? 0 : 1;
            if (predicted == actual) correct++;
        }
        
        Console.WriteLine($"Classification accuracy: {correct}/{inputs.Length} ({(float)correct / inputs.Length * 100:F1}%)");
    }

    public static void RegressionTest()
    {
        Console.WriteLine("\nTraining on multi-output regression (predicting sin and cos)...");
        
        var random = new Random(42);
        float[][] inputs = new float[30][];
        float[][] targets = new float[30][];
        
        for (int i = 0; i < 30; i++)
        {
            float x = (float)random.NextDouble() * MathF.PI * 2; // Range [0, 2π]
            inputs[i] = [x];
            targets[i] = [MathF.Sin(x), MathF.Cos(x)];
        }

        VectorFfNetwork network = new(
            1, 16, 2, 2,  
            ActivationFunctions.Tanh, 
            ActivationFunctions.Identity, 
            LossFunctions.MeanSquaredError);
        
        var inputStorage = inputs.Select(Operations.NewValue).ToArray();
        var targetStorage = targets.Select(Operations.NewValue).ToArray();
        
        float initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 3000, 0.01f);
        
        float finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nSample predictions:");
        float[] testAngles = [0, MathF.PI / 4, MathF.PI / 2, MathF.PI, 3 * MathF.PI / 2];
        foreach (var angle in testAngles)
        {
            float[] output = network.Forward(Operations.NewValue([angle])).ToHost();
            Console.WriteLine($"  x={angle:F4} → sin={output[0]:F4} (expected {Math.Sin(angle):F4}), cos={output[1]:F4} (expected {Math.Cos(angle):F4})");
        }
    }

    public static void RunAll()
    {
        OverfittingTest();
        OverfittingTestBatched();
        NonlinearFunctionTest();
        XorTest();
        IrisClassificationTest();
        RegressionTest();
    }
}