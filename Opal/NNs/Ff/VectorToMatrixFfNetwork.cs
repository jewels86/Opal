using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.NNs.Ff;

public class VectorToMatrixFfNetwork : FfNetwork<double[,], double[,], double[], double[,], double[,]>
{
    public VectorToMatrixFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[,]>? hiddenActivation = null,
        ActivationFunction<double[,]>? outputActivation = null,
        LossFunction<double[,]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "VectorToMatrixFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuMatrix,
            outputActivation ?? ActivationFunctions.ReLuMatrix,
            lossFunction ?? LossFunctions.MeanSquaredErrorMatrix,
            optimizer ?? new VectorToMatrixOptimizer(),
            new VectorToMatrixFfTensorOperations(),
            new StandardMatrixTensorOperations(),
            new StandardMatrixTensorOperations(),
            name)
    {
    }
}

public class VectorToMatrixOptimizer : IOptimizer<double[,], double[,]> {
    public double[,] UpdateBiases(double[,] biases, double[,] gradBiases, double learningRate) => Matrices.Subtract(biases, Matrices.Multiply(gradBiases, learningRate));
    public double[,] UpdateWeights(double[,] weights, double[,] gradWeights, double learningRate) => Matrices.Subtract(weights, Matrices.Multiply(gradWeights, learningRate));
}

public class VectorToMatrixFfTensorOperations : IFfTensorOperations<double[,], double[,], double[], double[,]>
{
    public double[,] Add(double[,] output, double[,] biases) => Matrices.Add(output, biases);
    public double[,] Apply(double[,] output, Func<double, double> activation) => Matrices.ApplyElementwise(output, activation);
    public double[,] DefaultBiases(int[] shape) => new double[shape[0], shape[1]];
    public double[,] DefaultOutput(int[] shape) => new double[shape[0], shape[1]];
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[] DefaultInput(int[] shape) => new double[shape[0]];
    public double[,] Multiply(double[,] weights, double[] input)
    {
        return Matrices.MultiplyMatrixByVectorAsColumn(weights, input);
    }
    public double[,] Multiply(double[,] a, double[,] b) => Matrices.Multiply(a, b);
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
