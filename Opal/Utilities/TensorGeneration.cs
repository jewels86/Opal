namespace Opal.Utilities;

public static class TensorGeneration
{
    public static readonly Random Random = new();
    #region Matrices
    public static Tensor<float[,]> GenerateMatrix(Func<int, int, float> generator, int rows, int columns)
    {
        var matrix = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
                matrix[i, j] = generator(i, j);
        return Operations.New(matrix, Operations.Fill(0, rows, columns));
    }

    public static Tensor<float[,]> GenerateMatrixParallel(Func<int, int, float> generator, int rows, int columns)
    {
        var matrix = new float[rows, columns];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
                matrix[i, j] = generator(i, j);
        });
        return Operations.New(matrix, Operations.Fill(0, rows, columns));
    }

    public static Tensor<float[,]> RandomMatrix(float min, float max, int rows, int cols) => 
        GenerateMatrix((_, _) => (float)Random.NextDouble() * (max - min) + min, rows, cols);
    public static Tensor<float[,]> RandomMatrixParallel(float min, float max, int rows, int cols) => 
        GenerateMatrixParallel((_, _) => (float)Random.NextDouble() * (max - min) + min, rows, cols);
    
    public static Tensor<float[,]> XavierMatrix(int rows, int columns)
    {
        var scale = MathF.Sqrt(2.0f / (rows + columns));
        return GenerateMatrix((_, _) => ((float)Random.NextDouble() * 2 - 1) * scale, rows, columns);
    }
    public static Tensor<float[,]> HeMatrix(int rows, int columns)
    {
        var scale = MathF.Sqrt(2.0f / rows);
        return GenerateMatrix((_, _) => Random.NextGaussian() * scale, rows, columns);
    }
    #endregion
    #region Vectors

    public static Tensor<float[]> GenerateVector(Func<int, float> generator, int size)
    {
        var vector = new float[size];
        for (int i = 0; i < size; i++) vector[i] = generator(i);
        return Operations.New(vector, Operations.Fill(0, size));
    }
    
    public static Tensor<float[]> RandomVector(float max, float min, int size)
    {
        var weights = new float[size];
        for (int i = 0; i < size; i++) weights[i] = (float)Random.NextDouble() * (max - min) + min;
        return Operations.New(weights, Operations.Fill(0, size));
    }
    
    public static Tensor<float[]> XavierVector(int size, int fanIn)
    {
        var scale = MathF.Sqrt(2.0f / (fanIn + size));
        return GenerateVector(_ => ((float)Random.NextDouble() * 2 - 1) * scale, size);
    }
    
    public static Tensor<float[]> HeVector(int size, int fanIn)
    {
        var scale = MathF.Sqrt(2.0f / fanIn);
        return GenerateVector(_ => Random.NextGaussian() * scale, size);
    }
    #endregion
    
    private static float NextGaussian(this Random random)
    {
        float u1 = 1.0f - (float)random.NextDouble();
        float u2 = 1.0f - (float)random.NextDouble();
        return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2);
    }
}