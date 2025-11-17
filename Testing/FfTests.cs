using Opal.Autograd;
using Opal.Mathematics;
using Opal.NNs.Ff;
using static Opal.Autograd.Operations;

namespace Testing;

public class FfTests
{
    public static void OverfittingTest()
    {
        Console.WriteLine($"Training a network to overfit a simple function...");
        double[] inputs = [0.5, -0.5];
        double[] targets = [1, -1];

        VectorFfNetwork network = new(
            1, 8, 1, 1,
            ActivationFunctions.IdentityVector, ActivationFunctions.IdentityVector, 
            LossFunctions.MeanSquaredErrorVector);
        
        var inputStorage = inputs.Select(x => NewDefaultVectorStorage([x])).ToArray();
        var targetStorage = targets.Select(x => NewDefaultVectorStorage([x])).ToArray();
        
        Console.WriteLine($"Initial loss: {network.EvaluateLoss(inputStorage, targetStorage)}");
        network.Train(inputStorage, targetStorage, 1000, 0.01);
        Console.WriteLine($"Evaluating the loss: {network.EvaluateLoss(inputStorage, targetStorage)}");
    }
    
    public static void NonlinearFunctionTest()
    {
        Console.WriteLine("\nTraining network on nonlinear function: f(x) = x^2...");
        
        // Generate training data for f(x) = x^2
        var random = new Random(42);
        double[] inputs = new double[20];
        double[] targets = new double[20];
        
        for (int i = 0; i < 20; i++)
        {
            inputs[i] = random.NextDouble() * 4 - 2; // Range [-2, 2]
            targets[i] = inputs[i] * inputs[i];
        }

        VectorFfNetwork network = new(
            1, 8, 1, 8,  
            ActivationFunctions.TanhVector, ActivationFunctions.IdentityVector, 
            LossFunctions.MeanSquaredErrorVector);
        
        Console.WriteLine($"Initial weights: {string.Join(", ", network.InputLayer.Weights.Value.ToHost())}");
        
        var inputStorage = inputs.Select(x => NewDefaultVectorStorage([x])).ToArray();
        var targetStorage = targets.Select(x => NewDefaultVectorStorage([x])).ToArray();
        
        double initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 2000, 0.01);
        
        Console.WriteLine($"Final weights: {string.Join(", ", network.InputLayer.Weights.Value.ToHost())}");
        
        double finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        Console.WriteLine($"Loss reduction: {(1 - finalLoss / initialLoss) * 100:F2}%");
        
        Console.WriteLine("\nTesting predictions:");
        double[] testInputs = [0.0, 1.0, -1.5, 2.0];
        foreach (var x in testInputs)
        {
            double prediction = network.Forward(NewCpuVectorStorage([x])).ToHost()[0];
            double expected = x * x;
            Console.WriteLine($"  f({x}) = {prediction:F4} (expected {expected:F4})");
        }
    }

    public static void XorTest()
    {
        Console.WriteLine("\nTraining network on XOR problem...");
        
        double[][] inputs = 
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ];
        
        double[][] targets = 
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ];

        VectorFfNetwork network = new(
            2, 4, 1, 1, 
            ActivationFunctions.TanhVector, 
            ActivationFunctions.SigmoidVector, 
            (predicted, actual) => LossFunctions.MeanSquaredErrorVector(predicted, actual));
        
        var inputStorage = inputs.Select(x => NewCpuVectorStorage(x)).ToArray();
        var targetStorage = targets.Select(x => NewCpuVectorStorage(x)).ToArray();
        
        double initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 5000, 0.5);
        
        double finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nXOR predictions:");
        foreach (var input in inputs)
        {
            double[] output = network.Forward(NewCpuVectorStorage(input)).ToHost();
            Console.WriteLine($"  [{input[0]}, {input[1]}] → {output[0]:F4}");
        }
    }

    public static void IrisClassificationTest()
    {
        Console.WriteLine("\nTraining on simplified Iris dataset (2 classes)...");
        
        // Simplified Iris dataset - just 2 features, 2 classes
        // Features: sepal length, sepal width (normalized)
        double[][] inputs = 
        [
            [0.22, 0.63], [0.17, 0.42], [0.11, 0.50], [0.08, 0.46], [0.19, 0.67],
            [0.31, 0.79], [0.19, 0.58], [0.20, 0.63], [0.11, 0.42], [0.25, 0.58],
            [0.69, 0.42], [0.56, 0.54], [0.61, 0.42], [0.53, 0.33], [0.56, 0.50],
            [0.67, 0.42], [0.61, 0.46], [0.64, 0.42], [0.69, 0.38], [0.56, 0.38]
        ];
        
        // One-hot encoded classes: [1,0] for setosa, [0,1] for versicolor
        double[][] targets = 
        [
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        ];

        VectorFfNetwork network = new(
            2, 8, 2, 2,  // 2 inputs, 8 hidden, 2 outputs, 2 hidden layers
            ActivationFunctions.ReLuVector, 
            ActivationFunctions.SoftmaxVector, 
            (predicted, actual) => LossFunctions.CrossEntropy(predicted, actual));
        
        var inputStorage = inputs.Select(x => NewCpuVectorStorage(x)).ToArray();
        var targetStorage = targets.Select(x => NewCpuVectorStorage(x)).ToArray();
        
        double initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 3000, 0.1);
        
        double finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        int correct = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double[] output = network.Forward(NewCpuVectorStorage(inputs[i])).ToHost();
            int predicted = output[0] > output[1] ? 0 : 1;
            int actual = targets[i][0] > targets[i][1] ? 0 : 1;
            if (predicted == actual) correct++;
        }
        
        Console.WriteLine($"Classification accuracy: {correct}/{inputs.Length} ({(double)correct / inputs.Length * 100:F1}%)");
    }

    public static void RegressionTest()
    {
        Console.WriteLine("\nTraining on multi-output regression (predicting sin and cos)...");
        
        var random = new Random(42);
        double[][] inputs = new double[30][];
        double[][] targets = new double[30][];
        
        for (int i = 0; i < 30; i++)
        {
            double x = random.NextDouble() * Math.PI * 2; // Range [0, 2π]
            inputs[i] = [x];
            targets[i] = [Math.Sin(x), Math.Cos(x)];
        }

        VectorFfNetwork network = new(
            1, 16, 2, 2,  
            ActivationFunctions.TanhVector, 
            ActivationFunctions.IdentityVector, 
            (predicted, actual) => LossFunctions.MeanSquaredErrorVector(predicted, actual));
        
        var inputStorage = inputs.Select(x => NewCpuVectorStorage(x)).ToArray();
        var targetStorage = targets.Select(x => NewCpuVectorStorage(x)).ToArray();
        
        double initialLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Initial loss: {initialLoss}");
        
        network.Train(inputStorage, targetStorage, 3000, 0.01);
        
        double finalLoss = network.EvaluateLoss(inputStorage, targetStorage);
        Console.WriteLine($"Final loss: {finalLoss}");
        
        Console.WriteLine("\nSample predictions:");
        double[] testAngles = [0.0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 2];
        foreach (var angle in testAngles)
        {
            double[] output = network.Forward(NewCpuVectorStorage([angle])).ToHost();
            Console.WriteLine($"  x={angle:F4} → sin={output[0]:F4} (expected {Math.Sin(angle):F4}), cos={output[1]:F4} (expected {Math.Cos(angle):F4})");
        }
    }

    public static void RunAll()
    {
        OverfittingTest();
        NonlinearFunctionTest();
        XorTest();
        IrisClassificationTest();
        RegressionTest();
    }
}