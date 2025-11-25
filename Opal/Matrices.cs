using Jewels.Lazulite;

namespace Opal;

public static partial class Operations
{
    public static float[,] Fill(float value, int cols, int rows)
    {
        var matrix = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
                matrix[i, j] = value;
        return matrix;
    }

    public static Tensor<float[,]> NewMatrix(float[,] matrix, float[,]? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null,
        int? aidx = null) =>
        new(new MatrixValue(matrix, aidx ?? DefaultAcceleratorIndex),
            new MatrixValue(gradient ?? Fill(0, matrix.GetLength(0), matrix.GetLength(1)), aidx ?? DefaultAcceleratorIndex),
            backwardAction, inputs);
    
    
}