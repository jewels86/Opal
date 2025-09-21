using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class VectorToMatrixFfNetwork : FfNetwork<double[,], double[,], double[], double[,], double[,]>
{
    public VectorToMatrixFfNetwork(
        int inputSize,
        int hiddenRows,
        int hiddenCols,
        int outputRows,
        int outputCols,
        int hiddenLayers,
        ActivationFunction<double[,]>? hiddenActivation = null,
        ActivationFunction<double[,]>? outputActivation = null,
        LossFunction<double[,]>? lossFunction = null,
        IFfOptimizer<double[,], double[,]>? optimizer = null,
        string name = "VectorToMatrixFfNetwork")
        : base(inputSize, hiddenRows * hiddenCols, outputRows * outputCols, hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuMatrix,
            outputActivation ?? ActivationFunctions.ReLuMatrix,
            lossFunction ?? LossFunctions.MeanSquaredErrorMatrix,
            optimizer ?? new VectorToMatrixFfOptimizer(),
            new VectorToMatrixFfTensorOperations(), new MatrixFfTensorOperations(), new MatrixFfTensorOperations(), name)
    {
    }
}

public class VectorToMatrixFfOptimizer : IFfOptimizer<double[,], double[,]> {
    public double[,] UpdateBiases(double[,] biases, double[,] gradBiases, double learningRate) => Matrices.Subtract(biases, Matrices.Multiply(gradBiases, learningRate));
    public double[,] UpdateWeights(double[,] weights, double[,] gradWeights, double learningRate) => Matrices.Subtract(weights, Matrices.Multiply(gradWeights, learningRate));
}

public class VectorToMatrixFfTensorOperations : IFfTensorOperations<double[,], double[,], double[], double[,]>
{
    public double[,] Add(double[,] output, double[,] biases) => Matrices.Add(output, biases);
    public double[,] Apply(double[,] output, Func<double, double> activation) => Matrices.ApplyElementwise(output, activation);
    public double[,] DefaultBiases(int size) => new double[size, size];
    public double[,] DefaultOutput(int size) => new double[size, size];
    public double[,] DefaultWeights(int rows, int cols) => Matrices.RandomMatrix(rows, cols);
    public double[] DefaultInput(int size) => new double[size];
    public double[,] Multiply(double[,] weights, double[] input)
    {
        return Matrices.MultiplyMatrixByVectorAsColumn(weights, input);
    }
    public double[] GradInput(double[,] weights, double[,] gradZ)
    {
        return Matrices.MultiplyMatrixTransposeByColumn(weights, gradZ);
    }
    public double[,] GradWeights(double[,] gradZ, double[] lastInput)
    {
        return Matrices.OuterProductColumnAndVector(gradZ, lastInput);
    }
    public double[,] GradBiases(double[,] gradZ) => gradZ;
}
