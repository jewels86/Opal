using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static partial class Operations
{
    public static float[,] Fill(float value, int rows, int cols)
    {
        var matrix = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
                matrix[i, j] = value;
        return matrix;
    }

    public static Tensor<float[,]> New(float[,] matrix, float[,]? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null, int? aidx = null) => new(
            new MatrixValue(matrix, aidx ?? DefaultAcceleratorIndex),
            new MatrixValue(gradient ?? Fill(0, matrix.GetLength(0), matrix.GetLength(1)), aidx ?? DefaultAcceleratorIndex),
            backwardAction, inputs);
    public static Tensor<float[,]> New(Value<float[,]> matrix, Value<float[,]> gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new(matrix, gradient, backwardAction, inputs);

    public static Value<float[,]> NewValue(float[,] matrix) => new MatrixValue(matrix, DefaultAcceleratorIndex);
    
    #region Kernels
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[]
        AddVectorToMatrixKernel { get; } = Compute.Load((Index1D i,
            ArrayView1D<float, Stride1D.Dense> matrix,
            ArrayView1D<float, Stride1D.Dense> vector,
            ArrayView1D<float, Stride1D.Dense> result, int n) => result[i] = matrix[i] + vector[i % n]);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>[] AddVectorToMatrixBackwardKernel { get; }
        = Compute.Load((Index1D col,
            ArrayView1D<float, Stride1D.Dense> vectorGrad,
            ArrayView1D<float, Stride1D.Dense> grad, int cols, int rows) =>
        {
            float sum = 0;
            for (int row = 0; row < rows; row++) sum += grad[row * cols + col];
            vectorGrad[col] += sum;
        });
    #endregion
    
    #region Multiplication
    public static Tensor<float[,]> MatrixMultiply(Tensor<float[,]> a, Tensor<float[,]> b, bool transposeA = false, bool transposeB = false)
    {
        var (aidx, m, k, n) = (a.AcceleratorIndex, a.Value.Shape[0], a.Value.Shape[1], b.Value.Shape[1]);
        var realM = transposeA ? a.Value.Shape[1] : a.Value.Shape[0];
        var realN = transposeB ? b.Value.Shape[0] : b.Value.Shape[1];
        
        var result = new MatrixValue(Compute.Get(aidx, realM * realN), [realM, realN]);
        Compute.MatrixMultiply(a.Value, b.Value, result, m, k, n, transposeA: transposeA, transposeB: transposeB, noCuBlas: true);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var (a0, a1) = (a.Value.Shape[0], a.Value.Shape[1]);
            var (b0, b1) = (b.Value.Shape[0], b.Value.Shape[1]);

            var gradA = Compute.Get(aidx, a0 * a1);
            var gradB = Compute.Get(aidx, b0 * b1);

            switch (transposeA, transposeB)
            {
                case (false, false):
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, b.Value.Data, gradA,
                        m, n, k,
                        transposeA: false, transposeB: true, noCuBlas: true
                    );
                    
                    Compute.MatrixMultiply(
                        a.Value.Data, tensor.Gradient.Data, gradB,
                        k, m, n,
                        transposeA: true, transposeB: false, noCuBlas: true
                    );
                    break;

                case (true, false):
                    Compute.MatrixMultiply(
                        b.Value.Data, tensor.Gradient.Data, gradA,
                        a1, n, m,
                        transposeA: false, transposeB: true, noCuBlas: true
                    );
                    
                    Compute.MatrixMultiply(
                        a.Value.Data, tensor.Gradient.Data, gradB,
                        k, m, n,
                        transposeA: false, transposeB: false, noCuBlas: true
                    );
                    break;

                case (false, true):
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, b.Value.Data, gradA,
                        a0, b0, a1,
                        transposeA: false, transposeB: false, noCuBlas: true
                    );
    
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, a.Value.Data, gradB,
                        b0, a0, a1,
                        transposeA: true, transposeB: false, noCuBlas: true
                    );
                    break;

                case (true, true):
                    Compute.MatrixMultiply(
                        b.Value.Data, tensor.Gradient.Data, gradA,
                        a1, n, m,
                        transposeA: true, transposeB: true
                    );
                    
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, a.Value.Data, gradB,
                        b1, m, a1,
                        transposeA: true, transposeB: true, noCuBlas: true
                    );
                    break;
            }

            Compute.Call(ElementwiseAccumulateKernels, gradA, a.Gradient);
            Compute.Call(ElementwiseAccumulateKernels, gradB, b.Gradient);
            Compute.Return(gradA, gradB);
        }
    }

    public static Tensor<float[]> MatrixVectorMultiply(Tensor<float[,]> matrix, Tensor<float[]> vector)
    {
        var (aidx, m, n) = (matrix.AcceleratorIndex, matrix.Value.Shape[0], matrix.Value.Shape[1]);
        var result = new VectorValue(Compute.Get(aidx, m));
        Compute.MatrixVectorMultiply(matrix.Value, vector.Value, result, m, n);
        return new(result, result.Zeros(), Backward, [matrix, vector]);

        void Backward(ITensor tensor)
        {
            var gradMatrix = Compute.Get(aidx, m * n);
            Compute.OuterProduct(tensor.Gradient.Data, vector.Value.Data, gradMatrix, m, n);
            Compute.Call(Compute.ElementwiseAddKernels, matrix.Gradient.Data, gradMatrix, matrix.Gradient.Data);
            
            var gradVector = Compute.Get(aidx, n);
            Compute.MatrixVectorMultiply(matrix.Value.Data, tensor.Gradient.Data, gradVector, m, n, transposeMatrix: true);
            Compute.Call(Compute.ElementwiseAddKernels, vector.Gradient.Data, gradVector, vector.Gradient.Data);
            
            Compute.Return(gradMatrix, gradVector);
        }
    }
    #endregion
    
    public static Tensor<float[,]> Add(Tensor<float[,]> matrix, Tensor<float[]> vector)
    {
        var (aidx, rows, cols) = (matrix.AcceleratorIndex, matrix.Value.Shape[0], matrix.Value.Shape[1]);
        var result = new MatrixValue(Compute.Get(aidx, rows * cols), [rows, cols]);
        Compute.Call(AddVectorToMatrixKernel, matrix.Value.Data, vector.Value.Data, result.Data, cols);
    
        return new(result, result.Zeros(), Backward, [matrix, vector]);

        void Backward(ITensor tensor)
        {
            Compute.Call(Compute.ElementwiseAddKernels, matrix.Gradient.Data, tensor.Gradient.Data, matrix.Gradient.Data);
            Compute.Call(AddVectorToMatrixBackwardKernel, vector.Gradient.Data, tensor.Gradient.Data, cols, rows);
        }
    }
}