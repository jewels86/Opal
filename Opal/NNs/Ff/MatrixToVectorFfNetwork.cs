using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class MatrixToVectorFfNetwork : FfNetwork<double[,], double[,], double[,], double[], double[]>
{
    public MatrixToVectorFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null,
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "MatrixToVectorFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector,
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.MeanSquaredErrorVector,
            optimizer ?? new StandardMatrixOptimizer(),
            new MatrixToVectorFfTensorOperations(),
            new MatrixToVectorHiddenFfTensorOperations(),
            new MatrixToVectorHiddenFfTensorOperations(),
            name)
    {
    }
}

public class MatrixToVectorFfTensorOperations : IFfTensorOperations<double[,], double[,], double[,], double[]>
{
    public double[] Add(double[] output, double[,] biases) => Vectors.Add(output, Matrices.Flatten(biases));
    public double[] Apply(double[] output, Func<double, double> activation) => Vectors.ApplyElementwise(output, activation);
    public double[,] DefaultBiases(int[] shape) => new double[shape[0], 1];
    public double[] DefaultOutput(int[] shape) => new double[shape[0]];
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0] * inputShape[1]);
    public double[,] DefaultInput(int[] shape) => new double[shape[0], shape[1]];
    public double[] Multiply(double[,] weights, double[,] input)
    {
        var inputFlat = Matrices.Flatten(input);
        return Matrices.Multiply(weights, inputFlat);
    }
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[,] GradBiases(double[] gradZ)
    {
        return Matrices.ToColumnVector(gradZ);
    }
    public double[,] GradInput(double[,] weights, double[] gradZ)
    {
        var gradInputVec = Matrices.Multiply(Matrices.Transpose(weights), gradZ);
        return Matrices.ToSquareMatrix(gradInputVec);
    }
    public double[,] GradWeights(double[] gradZ, double[,] lastInput)
    {
        var inputFlat = Matrices.Flatten(lastInput);
        return Matrices.OuterProduct(gradZ, inputFlat);
    }
}

public class MatrixToVectorHiddenFfTensorOperations : IFfTensorOperations<double[,], double[,], double[], double[]>
{
    public double[] Add(double[] output, double[,] biases) => Vectors.Add(output, Matrices.Flatten(biases));
    public double[] Apply(double[] output, Func<double, double> activation) => Vectors.ApplyElementwise(output, activation);
    public double[,] DefaultBiases(int[] shape) => new double[shape[0], 1];
    public double[] DefaultOutput(int[] shape) => new double[shape[0]];
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[] DefaultInput(int[] shape) => new double[shape[0]];
    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[,] GradBiases(double[] gradZ)
    {
        int size = gradZ.Length;
        var gradB = new double[size, 1];
        for (int i = 0; i < size; i++)
            gradB[i, 0] = gradZ[i];
        return gradB;
    }
    public double[] GradInput(double[,] weights, double[] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[,] GradWeights(double[] gradZ, double[] lastInput) => Matrices.OuterProduct(gradZ, lastInput);
}
