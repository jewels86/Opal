namespace Opal;

public static partial class Operations
{
    public static readonly Random Random = new();
    #region Tensor 3
    public static float[,,] Fill(float value, int x, int y, int z)
    {
        var tensor3 = new float[x, y, z];
        for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
        for (int k = 0; k < z; k++)
            tensor3[i, j, k] = value;
        return tensor3;
    }
    public static Tensor<float[,,]> GenerateTensor3(Func<int, int, int, float> generator, int x, int y, int z)
    {
        var tensor3 = new float[x, y, z];
        for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
        for (int k = 0; k < z; k++)
            tensor3[i, j, k] = generator(i, j, k);
        return New(tensor3, Fill(0, x, y, z));;
    }
    #endregion
    #region Matrices
    public static Tensor<float[,]> GenerateMatrix(Func<int, int, float> generator, int rows, int columns)
    {
        var matrix = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            matrix[i, j] = generator(i, j);
        return New(matrix, Fill(0, rows, columns));
    }

    public static Tensor<float[,]> GenerateMatrixParallel(Func<int, int, float> generator, int rows, int columns)
    {
        var matrix = new float[rows, columns];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < columns; j++)
                matrix[i, j] = generator(i, j);
        });
        return New(matrix, Fill(0, rows, columns));
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

    public static Tensor<float[,]> GenerateMatrix(Initialization init, int rows, int columns)
    {
        return init switch
        {
            Initialization.Random => RandomMatrix(1, -1, rows, columns),
            Initialization.Zeros => GenerateMatrix((_, _) => 0, rows, columns),
            Initialization.Ones => GenerateMatrix((_, _) => 1, rows, columns),
            Initialization.Xavier => XavierMatrix(rows, columns),
            Initialization.He => HeMatrix(rows, columns),
            _ => throw new ArgumentOutOfRangeException(nameof(init), init, null)
        };
    }
    #endregion
    #region Vectors
    public static Tensor<float[]> GenerateVector(Func<int, float> generator, int size)
    {
        var vector = new float[size];
        for (int i = 0; i < size; i++) vector[i] = generator(i);
        return New(vector, Fill(0, size));
    }

    public static Tensor<float[]> RandomVector(float max, float min, int size)
    {
        var weights = new float[size];
        for (int i = 0; i < size; i++) weights[i] = (float)Random.NextDouble() * (max - min) + min;
        return New(weights, Fill(0, size));
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

    public static Tensor<float[]> GenerateVector(Initialization init, int size, int? fanIn = null, int? max = null, int? min = null)
    {
        return init switch
        {
            Initialization.Random => RandomVector(max ?? 1, min ?? -1, size),
            Initialization.Zeros => GenerateVector(_ => 0, size),
            Initialization.Ones => GenerateVector(_ => 1, size),
            Initialization.Xavier => XavierVector(size, fanIn ?? size),
            Initialization.He => HeVector(size, fanIn ?? size),
            _ => throw new ArgumentOutOfRangeException(nameof(init), init, null)
        };
    }
    #endregion

    public static float NextGaussian(this Random random)
    {
        float u1 = 1.0f - (float)random.NextDouble();
        float u2 = 1.0f - (float)random.NextDouble();
        return MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2);
    }
}

public enum Initialization
{
    Random,
    Zeros,
    Ones,
    Xavier,
    He
}