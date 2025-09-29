using System.Data;

namespace Opal.Mathematics;

public static class Vectors
{
    private static readonly Random Random = new();
    
    public static double[] RandomVector(int size, double min = -1, double max = 1)
    {
        double[] vector = new double[size];
        double m = (max - min) + min;
        for (int i = 0; i < size; i++) 
            vector[i] = Random.NextDouble() * m;
        return vector;
    }
    
    #region Apply
    public static double[] ApplyElementwise(double[] input, Func<double, double> func)
    {
        int size = input.Length;
        var result = new double[size];
        for (int i = 0; i < size; i++)
            result[i] = func(input[i]);
        return result;
    }

    public static double[] ApplyElementwise(double[] input, Func<double, int, double> func)
    {
        int size = input.Length;
        var result = new double[size];
        for (int i = 0; i < size; i++)
            result[i] = func(input[i], i);
        return result;
    }
    #endregion
    
    #region Simple Operations
    public static double[] Add(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        return ApplyElementwise(a, (x, i) => x + b[i]);
    }
    public static double[] Subtract(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        return ApplyElementwise(a, (x, i) => x - b[i]);
    }
    public static double Dot(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }
    public static double[] Multiply(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        return ApplyElementwise(a, (x, i) => x * b[i]);
    }
    public static double[] Multiply(double[] a, double scalar) => ApplyElementwise(a, x => x * scalar);
    public static double[] Divide(double[] a, double scalar) => ApplyElementwise(a, x => x / scalar);
    public static double Sum(double[] a)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i];
        return sum;
    }
    #endregion
    #region Other Operations
    public static double[,] OuterProduct(double[] a, double[] b)
    {
        int rows = a.Length, cols = b.Length;
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = a[i] * b[j];
        return result;
    }
    public static double[] Concat(double[] a, double[] b) => a.Concat(b).ToArray();
    #endregion
}