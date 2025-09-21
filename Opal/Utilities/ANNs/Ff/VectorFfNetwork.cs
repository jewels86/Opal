using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class VectorFfNetwork : FfNetwork<double[,], double[], double[], double[], double[]>
{
    public VectorFfNetwork(int inputSize, int hiddenSize, int outputSize, int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null, 
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null, 
        IFfOptimizer<double[,], double[]>? optimizer = null,
        string name = "VectorFfNetwork")
        : base(inputSize, hiddenSize, outputSize, hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector, 
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.CrossEntropy, 
            optimizer ?? new StandardVectorFfOptimizer(),
            new VectorFfTensorOperations(), new VectorFfTensorOperations(), new VectorFfTensorOperations(), name)
    {
    }
}

public class VectorFfTensorOperations : IFfTensorOperations<double[,], double[], double[], double[]>
{
    public double[] Add(double[] output, double[] biases) => Vectors.Add(output, biases);

    public double[] Apply(double[] output, Func<double, double> activation) =>
        Vectors.ApplyElementwise(output, activation);

    public double[] DefaultBiases(int size) => new double[size];
    public double[] DefaultOutput(int size) => new double[size];
    public double[,] DefaultWeights(int rows, int cols) => Matrices.RandomMatrix(rows, cols);
    public double[] DefaultInput(int size) => new double[size];

    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] GradBiases(double[] gradZ) => gradZ;
    public double[] GradInput(double[,] weights, double[] gradZ)
    {
        int rows = weights.GetLength(0), cols = weights.GetLength(1);
        if (rows != gradZ.Length)
            throw new ArgumentException("Weights rows must match gradZ size.");
        
        return Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    }

    public double[,] GradWeights(double[] gradZ, double[] lastInput) => Vectors.OuterProduct(lastInput, gradZ);
}

public class StandardVectorFfOptimizer : IFfOptimizer<double[,], double[]>
{
    public double[] UpdateBiases(double[] biases, double[] gradBiases, double learningRate) => Vectors.Subtract(biases, Vectors.Multiply(gradBiases, learningRate));
    public double[,] UpdateWeights(double[,] weights, double[,] gradWeights, double learningRate) => Matrices.Subtract(weights, Matrices.Multiply(gradWeights, learningRate));
}