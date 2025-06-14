using Opal.Utilities.ANNs;

namespace Opal.Utilities;

public static class MathFunctions
{
    public static double[] ReLu(double[] x) => x.Select(v => Math.Max(0, v)).ToArray();
    public static double[] ReLuDerivative(double[] x) => x.Select(v => v > 0 ? 1.0 : 0.0).ToArray();

    public static double[,] InitializeMatrix(int rows, int cols, Func<int, int, double> initializer)
    {
        double[,] matrix = new double[rows,cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = initializer(i, j);
            }
        }

        return matrix;
    }

    public static double[,] RandomMatrix(int rows, int cols)
    {
        return InitializeMatrix(rows, cols, (i, j) => Random.Shared.NextDouble() * 2 - 1);
    }
    public static double[,] ZeroMatrix(int rows, int cols)
    {
        return InitializeMatrix(rows, cols, (i, j) => 0.0);
    }
    
    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    public static double[] Sigmoid(double[] x) => x.Select(Sigmoid).ToArray();
    public static double Tanh(double x) => Math.Tanh(x);
    public static double[] Tanh(double[] x) => x.Select(Tanh).ToArray();
    
    public static double SigmoidDerivative(double x) => Sigmoid(x) * (1 - Sigmoid(x));
    public static double[] SigmoidDerivative(double[] x) => x.Select(SigmoidDerivative).ToArray();
    public static double TanhDerivative(double x) => 1.0 - Math.Pow(Tanh(x), 2);
    public static double[] TanhDerivative(double[] x) => x.Select(TanhDerivative).ToArray();

    public static double[] Apply(ILayer.ActivationFunction fn, double[] x)
    {
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            y[i] = fn(x[i]);
        return y;
    }
    public static double[] Add(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x + y).ToArray();
    }
    public static double[] Subtract(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x - y).ToArray();
    }
    public static double[] Multiply(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x * y).ToArray();
    }
    public static double[] Multiply(double[] a, double scalar)
    {
        return a.Select(x => x * scalar).ToArray();
    }

    public static double[] Multiply(double[,] matrix, double[] vector)
    {
        if (matrix.GetLength(1) != vector.Length) throw new ArgumentException("Matrix columns must match vector length.");
        double[] result = new double[matrix.GetLength(0)];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            result[i] = 0;
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }
        return result;
    }
    public static double[] Divide(double[] a, double scalar)
    {
        if (scalar == 0) throw new DivideByZeroException("Cannot divide by zero.");
        return a.Select(x => x / scalar).ToArray();
    }
    public static double[] Dot(double[] a, double[] b) 
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return new double[] { a.Zip(b, (x, y) => x * y).Sum() };
    }
    
    public static double[,] GetBatchSample(double[,,] inputSequence, int batchIdx)
    {
        int time = inputSequence.GetLength(1);
        int inputSize = inputSequence.GetLength(2);
        double[,] sample = new double[time, inputSize];
        for (int t = 0; t < time; t++)
        for (int i = 0; i < inputSize; i++)
            sample[t, i] = inputSequence[batchIdx, t, i];
        return sample;
    }
    public static double[] GetInputFromSample(double[,] sample, int timeStep)
    {
        int inputSize = sample.GetLength(1);
        double[] input = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
            input[i] = sample[timeStep, i];
        return input;
    }
}