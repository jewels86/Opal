namespace Opal.Mathematics;

public static class Tensors
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
    public static double[,] RandomMatrix(int rows, int cols, double min = -1, double max = 1)
    {
        double[,] matrix = new double[rows, cols];
        double m = max - min;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = Random.NextDouble() * m + min;
        return matrix;
    }
}

public interface ITensorOperations<T> where T : notnull
{
    public T Add(T a, T b);
    public T Subtract(T a, T b);
    public T Multiply(T a, T b);
    public T Apply(T tensor, Func<double, double> func);
}