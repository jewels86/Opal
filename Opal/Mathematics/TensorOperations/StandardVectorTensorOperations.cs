using Opal.NNs.Ff;
using Opal.NNs.Lstm.Attention;

namespace Opal.Mathematics.TensorOperations;

public class StandardVectorTensorOperations :
    IFfTensorOperations<double[,], double[], double[], double[]>,
    ILstmAttentionTensorOperations<double[,], double[], double[]>
{
    public double[] Add(double[] a, double[] b) => Vectors.Add(a, b);
    public double Dot(double[] a, double[] b) => Vectors.Dot(a, b);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);
    public double[] Multiply(double[] a, double scalar) => Vectors.Multiply(a, scalar);
    public double[] Add(double[] a, double[] b, double[] c, double[] d) => Vectors.Add(Vectors.Add(a, b), Vectors.Add(c, d));
    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] Multiply(double[] a, double[] b, double[] c) => Vectors.Multiply(Vectors.Multiply(a, b), c);
    public double[] Concat(double[] a, double[] b) => Vectors.Concat(a, b);
    public double[] Apply(double[] output, Func<double, double> activation) => Vectors.ApplyElementwise(output, activation);

    public double[] DefaultBiases(int[] outputShape) => new double[outputShape[0]];
    public double[,] DefaultWeights(int[] inputShape, int[] outputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0], -0.5, 0.5);
    public double[] DefaultCell(int[] shape) => new double[shape[0]];
    public double[] DefaultInput(int[] inputShape) => new double[inputShape[0]];
    public double[] DefaultOutput(int[] outputsShape) => new double[outputsShape[0]];
    public double[] DefaultState(int[] shape) => new double[shape[0]];
    
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

    public double[] GradBiases(double[] gradZ) => gradZ;
    public double[] GradInput(double[,] weights, double[] gradZ) => Multiply(Matrices.Transpose(weights), gradZ);
    public double[,] GradWeights(double[] gradZ, double[] lastInput) 
    {
        int rows = gradZ.Length;
        int cols = lastInput.Length;
        double[,] gradWeights = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            gradWeights[i, j] = gradZ[i] * lastInput[j];
        return gradWeights;
    }

    public double[] WeightedSum(double[][] tensors, double[] weights)
    {
        if (tensors.Length != weights.Length)
            throw new ArgumentException("Number of tensors must match number of weights.");
        int size = tensors[0].Length;
        var result = new double[size];
        for (int i = 0; i < tensors.Length; i++)
        {
            if (tensors[i].Length != size)
                throw new ArgumentException("All tensors must have the same size.");
            for (int j = 0; j < size; j++)
                result[j] += tensors[i][j] * weights[i];
        }
        return result;
    }
}