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
}