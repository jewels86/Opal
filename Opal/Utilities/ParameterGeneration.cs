using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.Utilities;

public static class ParameterGeneration
{
    public static readonly Random Random = new();
    public static MatrixTensor GenerateMatrix(Func<int, int, double> generator, int rows, int columns)
    {
        var weights = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
                weights[i, j] = generator(i, j);
        return Operations.NewMatrix(weights, Matrices.Zeros(rows, columns));
    }

    public static VectorTensor GenerateVector(Func<int, double> generator, int size)
    {
        var weights = new double[size];
        for (int i = 0; i < size; i++) weights[i] = generator(i);
        return Operations.NewVector(weights, Vectors.Zeros(size));
    }

    public static MatrixTensor RandomMatrix(double max, double min, int rows, int columns)
    {
        var weights = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            weights[i, j] = Random.NextDouble() * (max - min) + min;
        return Operations.NewMatrix(weights, Matrices.Zeros(rows, columns));
    }
    
    public static VectorTensor RandomVector(double max, double min, int size)
    {
        var weights = new double[size];
        for (int i = 0; i < size; i++) weights[i] = Random.NextDouble() * (max - min) + min;
        return Operations.NewVector(weights, Vectors.Zeros(size));   
    }
}