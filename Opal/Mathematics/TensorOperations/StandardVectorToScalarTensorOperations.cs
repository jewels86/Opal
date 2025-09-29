using System;
using Opal.NNs.Ff;

namespace Opal.Mathematics.TensorOperations;

public class StandardVectorToScalarTensorOperations : IFfTensorOperations<double[,], double[], double[], double>
{
    public double Multiply(double[,] weights, double[] input)
    {
        var outputVec = Matrices.Multiply(weights, input);
        return Vectors.Sum(outputVec);
    }

    public double Multiply(double a, double b) => a * b;

    public double Add(double output, double[] biases)
    {
        return output + Vectors.Sum(biases);
    }

    public double Apply(double output, Func<double, double> activation) => activation(output);

    public double[] GradInput(double[,] weights, double gradZ)
    {
        var transposed = Matrices.Transpose(weights);
        var gradInput = new double[transposed.GetLength(0)];
        for (int i = 0; i < gradInput.Length; i++)
        {
            double sum = 0;
            for (int j = 0; j < transposed.GetLength(1); j++)
                sum += transposed[i, j] * gradZ;
            gradInput[i] = sum;
        }
        return gradInput;
    }

    public double[,] GradWeights(double gradZ, double[] lastInput)
    {
        int rows = 1;
        int cols = lastInput.Length;
        double[,] gradWeights = new double[rows, cols];
        for (int j = 0; j < cols; j++)
            gradWeights[0, j] = gradZ * lastInput[j];
        return gradWeights;
    }

    public double[] GradBiases(double gradZ)
    {

        return [gradZ];
    }

    public double[] DefaultInput(int[] inputShape) => new double[inputShape[0]];

    public double DefaultOutput(int[] outputsShape) => 0.0;

    public double[,] DefaultWeights(int[] outputShape, int[] inputShape)
    {
        return Matrices.RandomMatrix(outputShape[0], inputShape[0], -0.5, 0.5);
    }

    public double[] DefaultBiases(int[] outputShape)
    {
        return new double[outputShape[0]];
    }
}