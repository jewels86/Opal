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
    
    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    public static double[] Sigmoid(double[] x) => x.Select(Sigmoid).ToArray();
    public static double Tanh(double x) => Math.Tanh(x);
    public static double[] Tanh(double[] x) => x.Select(Tanh).ToArray();

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
}