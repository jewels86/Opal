using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Lstm;

public class VectorLstmNetwork : LstmNetwork<double[,], double[], double[]>
{
    public VectorLstmNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[]> sigmoidActivation, ActivationFunction<double[]> tanhActivation,
        LossFunction<double[][]> lossFunction, IOptimizer<double[,], double[]> optimizer,
        string name = "vector lstm network")
        : base(inputShape, hiddenShape, outputShape, hiddenLayers, sigmoidActivation, tanhActivation, lossFunction,
            optimizer, new VectorLstmTensorOperations(), name)
    {
    }
}

public class VectorLstmTensorOperations : ILstmTensorOperations<double[,], double[], double[]>
{
    public double[] Add(double[] a, double[] b) => Vectors.Add(a, b);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[] Add(double[] a, double[] b, double[] c, double[] d) => Vectors.Add(Vectors.Add(a, b), Vectors.Add(c, d));
    public double[] Concat(double[] a, double[] b) => Vectors.Concat(a, b);
    public double[] Multiply(double[,] a, double[] b) => Matrices.Multiply(a, b);
    public double[] Multiply(double[] a, double[] b, double[] c) => Vectors.Multiply(Vectors.Multiply(a, b), c);
    

    public double[] DefaultState(int[] outputsShape) => new double[outputsShape[0]];
    public double[] DefaultBiases(int[] outputShape) => new double[outputShape[0]];
    public double[,] DefaultWeights(int[] inputShape, int[] outputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[] DefaultCell(int[] shape) => new double[shape[0]];

    public void UpdateAccumulatedBiases(double[] biases, double[] dForgetGatePre)
    {
        for (int i = 0; i < biases.Length; i++)
            biases[i] += dForgetGatePre[i];
    }
    public void UpdateAccumulatedWeights(double[,] weights, double[] inputs, double[] dForgetGatePre)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
            for (int j = 0; j < weights.GetLength(1); j++)
                weights[i, j] += inputs[j] * dForgetGatePre[i];
    }
}