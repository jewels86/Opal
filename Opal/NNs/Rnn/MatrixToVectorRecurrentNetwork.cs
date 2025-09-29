using Opal.Mathematics;

namespace Opal.NNs.Rnn;

public class MatrixToVectorRecurrentNetwork : RecurrentNetwork<double[,], double[,], double[], double[,], double[], double[]>
{
    public MatrixToVectorRecurrentNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null,
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "MatrixToVectorRecurrentNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector,
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.MeanSquaredErrorVector,
            optimizer ?? new StandardMatrixOptimizer(),
            new MatrixToVectorRecurrentTensorOperations(),
            new MatrixToVectorHiddenRecurrentTensorOperations(),
            new MatrixToVectorHiddenRecurrentTensorOperations(),
            name)
    {
    }
}

public class MatrixToVectorRecurrentTensorOperations : IRecurrentTensorOperations<double[,], double[,], double[,], double[], double[]>
{
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0] * inputShape[1]);
    public double[,] DefaultBiases(int[] outputShape) => new double[outputShape[0], 1];
    public double[] DefaultState(int[] outputsShape) => new double[outputsShape[0]];

    public double[] Add(double[] a, double[,] b) => Vectors.Add(a, Matrices.Flatten(b));
    public double[] Add(double[] a, double[] b) => Vectors.Add(a, b);
    public double[,] Add(double[,] a, double[,] b) => Matrices.Add(a, b);
    public double[,] Add(double[,] a, double[] b) => Matrices.Add(a, Matrices.ToColumnVector(b));
    public double[] Multiply(double[,] weights, double[,] input)
    {
        var inputFlat = Matrices.Flatten(input);
        return Matrices.Multiply(weights, inputFlat);
    }
    public double[] Multiply(double[,] weights, double[] state) => Matrices.Multiply(weights, state);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[,] GradInputWeights(double[] gradZ, double[,] input)
    {
        var inputFlat = Matrices.Flatten(input);
        return Matrices.OuterProduct(gradZ, inputFlat);
    }
    public double[,] GradRecurrentWeights(double[] gradZ, double[] state) => Matrices.OuterProduct(gradZ, state);
    public double[,] GradBiases(double[] gradZ) => Matrices.ToColumnVector(gradZ);
    public double[] GradOutput(double[,] weights, double[] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[,] GradInput(double[,] weights, double[] gradZ)
    {
        var gradInputVec = Matrices.Multiply(Matrices.Transpose(weights), gradZ);
        return Matrices.ToSquareMatrix(gradInputVec);
    }
    public double[] UpdateState(double[] output) => output;
}

public class MatrixToVectorHiddenRecurrentTensorOperations : IRecurrentTensorOperations<double[,], double[,], double[], double[], double[]>
{
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[,] DefaultBiases(int[] outputShape) => new double[outputShape[0], 1];
    public double[] DefaultState(int[] outputsShape) => new double[outputsShape[0]];

    public double[] Add(double[] a, double[,] b) => Vectors.Add(a, Matrices.Flatten(b));
    public double[] Add(double[] a, double[] b) => Vectors.Add(a, b);
    public double[,] Add(double[,] a, double[,] b) => Matrices.Add(a, b);
    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[,] GradInputWeights(double[] gradZ, double[] input) => Matrices.OuterProduct(gradZ, input);
    public double[,] GradRecurrentWeights(double[] gradZ, double[] state) => Matrices.OuterProduct(gradZ, state);
    public double[,] GradBiases(double[] gradZ) => Matrices.ToColumnVector(gradZ);
    public double[] GradOutput(double[,] weights, double[] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[] GradInput(double[,] weights, double[] gradZ) => Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    public double[] UpdateState(double[] output) => output;
}

