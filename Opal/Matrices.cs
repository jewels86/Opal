using ILGPU;
using ILGPU.Runtime;
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
    
    #region Kernels
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[]
        AddVectorToMatrixKernel { get; } = Compute.Load((Index1D i,
            ArrayView1D<float, Stride1D.Dense> matrix,
            ArrayView1D<float, Stride1D.Dense> vector,
            ArrayView1D<float, Stride1D.Dense> result, int n) => result[i] = matrix[i] + vector[i % n]);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] AddVectorToMatrixBackwardKernel { get; }
        = Compute.Load((Index1D col,
            ArrayView1D<float, Stride1D.Dense> grad,
            ArrayView1D<float, Stride1D.Dense> vectorGrad, int cols, int rows) =>
        {
            float sum = 0;
            for (int row = 0; row < rows; row++) sum += grad[row * cols + col];
            vectorGrad[col] += sum;
        });
    #endregion
    
    #region Multiplication
    public static Tensor<float[,]> MatrixMultiply(Tensor<float[,]> a, Tensor<float[,]> b, bool disposeA = true, bool disposeB = true)
    {
        var aidx = a.AcceleratorIndex;
        var result = new MatrixValue(Compute.Get(aidx, a.Value.Shape[0] * b.Value.Shape[1]), [a.Value.Shape[0], b.Value.Shape[1]]);
        Compute.MatrixMultiply(a.Value, b.Value, result, a.Value.Shape[0], b.Value.Shape[0], b.Value.Shape[1]);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var gradA = Compute.Get(aidx, a.Value.TotalSize);
            Compute.MatrixMultiply(tensor.Gradient.Data, b.Value.Data, gradA, 
                a.Value.Shape[0], b.Value.Shape[0], b.Value.Shape[1], transposeB: true);
            Compute.Call(Compute.ElementwiseAddKernels, a.Gradient.Data, gradA, a.Gradient.Data);

            var gradB = Compute.Get(aidx, b.Value.TotalSize);
            Compute.MatrixMultiply(a.Value.Data, tensor.Gradient.Data, gradB,
                a.Value.Shape[1], b.Value.Shape[1], a.Value.Shape[0], transposeA: true);
            Compute.Call(Compute.ElementwiseAddKernels, b.Gradient.Data, gradB, b.Gradient.Data);
        
            Compute.Return(gradA, gradB);
        }
    }

    public static Tensor<float[]> Multiply(Tensor<float[,]> matrix, Tensor<float[]> vector, bool disposeA = true, bool disposeB = true)
    {
        var aidx = matrix.AcceleratorIndex;
        var m = matrix.Value.Shape[0];
        var n = matrix.Value.Shape[1];
    
        var result = new VectorValue(Compute.Get(aidx, m));
        Compute.MatrixVectorMultiply(matrix.Value, vector.Value, result, m, n);
        return new(result, result.Zeros(), Backward, [matrix, vector]);

        void Backward(ITensor tensor)
        {
            var gradMatrix = Compute.GetTemp(aidx, m * n);
            Compute.MatrixMultiply(tensor.Gradient.Data, vector.Value.Data, gradMatrix, m, 1, n);
            Compute.Call(Compute.ElementwiseAddKernels, matrix.Gradient.Data, gradMatrix, matrix.Gradient.Data);

            var gradVector = Compute.GetTemp(aidx, n);
            Compute.MatrixVectorMultiply(matrix.Value.Data, tensor.Gradient.Data, gradVector, m, n, transposeMatrix: true);
            Compute.Call(Compute.ElementwiseAddKernels, vector.Gradient.Data, gradVector, vector.Gradient.Data);
            
            if (disposeA) matrix.Dispose();
            if (disposeB) vector.Dispose();
        }
    }
    #endregion
}