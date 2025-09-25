using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class MatrixFfNetwork : FfNetwork<double[,], double[,], double[,], double[,], double[,]>
{
    public MatrixFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[,]>? hiddenActivation = null,
        ActivationFunction<double[,]>? outputActivation = null,
        LossFunction<double[,]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "MatrixFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuMatrix,
            outputActivation ?? ActivationFunctions.ReLuMatrix,
            lossFunction ?? LossFunctions.MeanSquaredErrorMatrix,
            optimizer ?? new StandardMatrixOptimizer(),
            new MatrixFfTensorOperations(),
            new MatrixFfTensorOperations(),
            new MatrixFfTensorOperations(),
            name)
    {
    }
}

public class MatrixFfTensorOperations : IFfTensorOperations<double[,], double[,], double[,], double[,]>
{
    public double[,] Add(double[,] output, double[,] biases) => Matrices.Add(output, biases);

    public double[,] Apply(double[,] output, Func<double, double> activation) => Matrices.ApplyElementwise(output, activation);

    public double[,] DefaultBiases(int[] shape) => new double[shape[0], shape[1]];

    public double[,] DefaultOutput(int[] shape) => new double[shape[0], shape[1]];
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[1]);
    public double[,] DefaultInput(int[] shape) => new double[shape[0], shape[1]];

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
