using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class MatrixFfNetwork : FfNetwork<double[,], double[,], double[,], double[,], double[,]>
{
    public MatrixFfNetwork(int inputRows, int inputCols, int hiddenRows, int hiddenCols, int outputRows, int outputCols, int hiddenLayers,
        ActivationFunction<double[,]>? hiddenActivation = null,
        ActivationFunction<double[,]>? outputActivation = null,
        LossFunction<double[,]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "MatrixFfNetwork")
        : base(inputRows * inputCols, hiddenRows * hiddenCols, outputRows * outputCols, hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuMatrix,
            outputActivation ?? ActivationFunctions.ReLuMatrix,
            lossFunction ?? LossFunctions.MeanSquaredErrorMatrix,
            optimizer ?? new StandardMatrixOptimizer(),
            new MatrixFfTensorOperations(), new MatrixFfTensorOperations(), new MatrixFfTensorOperations(), name)
    {
    }
}

public class MatrixFfTensorOperations : IFfTensorOperations<double[,], double[,], double[,], double[,]>
{
    public double[,] Add(double[,] output, double[,] biases) => Matrices.Add(output, biases);

    public double[,] Apply(double[,] output, Func<double, double> activation) => Matrices.ApplyElementwise(output, activation);

    public double[,] DefaultBiases(int size) => new double[size, size];

    public double[,] DefaultOutput(int size) => new double[size, size];
    public double[,] DefaultWeights(int rows, int cols) => Matrices.RandomMatrix(rows, cols);
    public double[,] DefaultInput(int size) => new double[size, size];

    public double[,] Multiply(double[,] weights, double[,] input)
    {
        return Matrices.Multiply(weights, input);
    }

    public double[,] GradBiases(double[,] gradZ) => gradZ;

    public double[,] GradInput(double[,] weights, double[,] gradZ)
    {
        return Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    }

    public double[,] GradWeights(double[,] gradZ, double[,] lastInput)
    {
        return Matrices.OuterProduct(Matrices.Flatten(gradZ), Matrices.Flatten(lastInput));
    }
}

