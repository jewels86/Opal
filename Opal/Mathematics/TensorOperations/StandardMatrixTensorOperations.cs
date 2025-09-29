using Opal.NNs.Ff;
using Opal.NNs.Lstm.Attention;

namespace Opal.Mathematics.TensorOperations;

public class StandardMatrixTensorOperations :
    IFfTensorOperations<double[,], double[,], double[,], double[,]>,
    ILstmAttentionTensorOperations<double[,], double[,], double[,]>
{
    public double[,] Multiply(double[,] weights, double[,] input) => Matrices.Multiply(weights, input);
    public double[,] Multiply(double[,] a, double b) => Matrices.Multiply(a, b);
    public double[,] Multiply(double[,] a, double[,] b, double[,] c) => Matrices.Multiply(Matrices.Multiply(a, b), c);
    public double[,] Add(double[,] output, double[,] biases) => Matrices.Add(output, biases);
    public double[,] Add(double[,] a, double[,] b, double[,] c, double[,] d) => Matrices.Add(Matrices.Add(a, b), Matrices.Add(c, d));
    public double[,] Concat(double[,] a, double[,] b) => Matrices.Concat(a, b);
    public double[,] Apply(double[,] output, Func<double, double> activation) => Matrices.ApplyElementwise(output, activation);
    public double Dot(double[,] a, double[,] b) => Matrices.Dot(a, b);
    public double[,] WeightedSum(double[][,] tensors, double[] weights)
    {
        if (tensors.Length != weights.Length)
            throw new ArgumentException("Number of tensors must match number of weights.");
        int rows = tensors[0].GetLength(0);
        int cols = tensors[0].GetLength(1);
        var result = new double[rows, cols];
        for (int i = 0; i < tensors.Length; i++)
        {
            if (tensors[i].GetLength(0) != rows || tensors[i].GetLength(1) != cols)
                throw new ArgumentException("All tensors must have the same dimensions.");
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r, c] += tensors[i][r, c] * weights[i];
        }
        return result;
    }

    public void UpdateAccumulatedBiases(double[,] biases, double[,] dForgetGatePre)
    {
        for (int i = 0; i < biases.GetLength(0); i++)
        for (int j = 0; j < biases.GetLength(1); j++)
            biases[i, j] += dForgetGatePre[i, j];
    }

    public void UpdateAccumulatedWeights(double[,] weights, double[,] dForgetGatePre, double[,] concat)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
        for (int j = 0; j < weights.GetLength(1); j++)
            weights[i, j] += concat[j, 0] * dForgetGatePre[i, 0];
    }

    public double[,] DefaultBiases(int[] outputShape) => new double[outputShape[0], outputShape[1]];
    public double[,] DefaultWeights(int[] inputShape, int[] outputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0], -0.5, 0.5);
    public double[,] DefaultCell(int[] shape) => new double[shape[0], shape[1]];
    public double[,] DefaultInput(int[] inputShape) => new double[inputShape[0], inputShape[1]];
    public double[,] DefaultOutput(int[] outputsShape) => new double[outputsShape[0], outputsShape[1]];
    public double[,] DefaultState(int[] shape) => new double[shape[0], shape[1]];
    public double[,] GradBiases(double[,] gradZ) => gradZ;
    public double[,] GradInput(double[,] weights, double[,] gradZ) => Multiply(Matrices.Transpose(weights), gradZ);

    public double[,] GradWeights(double[,] gradZ, double[,] lastInput)
    {
        int rows = gradZ.GetLength(0);
        int cols = lastInput.GetLength(0);
        double[,] gradWeights = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            gradWeights[i, j] = gradZ[i, 0] * lastInput[j, 0];
        return gradWeights;
    }
}