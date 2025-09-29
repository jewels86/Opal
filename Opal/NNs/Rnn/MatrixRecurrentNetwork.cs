using Opal.Mathematics;

namespace Opal.NNs.Rnn;

public class MatrixRecurrentNetwork : RecurrentNetwork<double[,], double[,], double[,], double[,], double[,], double[,]>
{
    public MatrixRecurrentNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[,]> hiddenActivation,
        ActivationFunction<double[,]> outputActivation,
        LossFunction<double[,]> lossFunction,
        IOptimizer<double[,], double[,]> optimizer,
        string name = "MatrixRecurrentNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation,
            outputActivation,
            lossFunction,
            optimizer,
            new RecurrentMatrixTensorOperations(),
            new RecurrentMatrixTensorOperations(),
            new RecurrentMatrixTensorOperations(),
            name)
    {
    }
}

public class RecurrentMatrixTensorOperations : IRecurrentTensorOperations<double[,], double[,], double[,], double[,], double[,]>
{
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[1]);
    public double[,] DefaultBiases(int[] outputShape) => new double[outputShape[0], outputShape[1]];
    public double[,] DefaultState(int[] outputsShape) => new double[outputsShape[0], outputsShape[1]];

    public double[,] Add(double[,] a, double[,] b) => Matrices.Add(a, b);
    public double[,] Multiply(double[,] weights, double[,] input) => Matrices.Multiply(weights, input);
    public double[,] GradInputWeights(double[,] gradZ, double[,] input) => Matrices.OuterProduct(Matrices.Flatten(gradZ), Matrices.Flatten(input));
    public double[,] GradRecurrentWeights(double[,] gradZ, double[,] state) => Matrices.OuterProduct(Matrices.Flatten(gradZ), Matrices.Flatten(state));
    public double[,] GradBiases(double[,] gradZ) => gradZ;
    public double[,] GradOutput(double[,] weights, double[,] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[,] GradInput(double[,] weights, double[,] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[,] UpdateState(double[,] output) => output;
}