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
    
    public static Tensor<float[]> Multiply(Tensor<float[,]> a, Tensor<float[]> b)
    {
        var aidx = a.AcceleratorIndex;
        var result = new VectorValue(Compute.Get(aidx, a.Value.Shape[0] * b.Value.Shape[1]));
        Compute.MatrixMultiply(a.Value, b.Value, result, a.Value.Shape[0], b.Value.Shape[1], b.Value.Shape[0]);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var gradA = Compute.Get(aidx, a.Value.TotalSize);
            Compute.MatrixMultiply(tensor.Gradient.Data, b.Value.Data, gradA, 
                a.Value.Shape[0], b.Value.Shape[0], b.Value.Shape[1], transposeB: true);
            Compute.Call(aidx, Compute.ElementwiseAddKernels, a.Gradient.Data, gradA, a.Gradient.Data);

            var gradB = Compute.Get(aidx, b.Value.TotalSize);
            Compute.MatrixMultiply(a.Value.Data, tensor.Gradient.Data, gradB,
                a.Value.Shape[1], b.Value.Shape[1], a.Value.Shape[0], transposeA: true);
            Compute.Call(aidx, Compute.ElementwiseAddKernels, b.Gradient.Data, gradB, b.Gradient.Data);
        
            Compute.Return(gradA, gradB);
        }
    }
}