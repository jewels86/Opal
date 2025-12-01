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
        var (aidx, a0, a1, b0, b1) = (a.AcceleratorIndex, a.Value.Shape[0], a.Value.Shape[1], b.Value.Shape[0], b.Value.Shape[1]);
        var m = transposeA ? a1 :  a0;
        var n = transposeB ? b0 : b1;
        
        var result = new MatrixValue(Compute.Get(aidx, m * n), [m, n]);
        Compute.MatrixMultiply(a.Value, b.Value, result, a0, a1, b0, b1, transposeA: transposeA, transposeB: transposeB);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {

            var gradA = Compute.Get(aidx, a0 * a1);
            var gradB = Compute.Get(aidx, b0 * b1);

            switch (transposeA, transposeB)
            {
                case (false, false):
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, b.Value.Data, gradA,
                        a0, b1, b0, b1,
                        transposeA: false, transposeB: true // transpose b
                    ); // d/da axb = L'(O) x b^T, m = a0, k = b1, n = a1
                    // grad is a0xb1, b is b0xb1 (transposed is b1xb0)
    
                    Compute.MatrixMultiply(
                        a.Value.Data, tensor.Gradient.Data, gradB,
                        a0, a1, a0, b1,
                        transposeA: true, transposeB: false // transpose a
                    ); // d/db axb = a^T x L'(O), m = b0, k = a0, n = b1
                    // a is a0xa1 (transposed is a1xa0), grad is a0xb1
                    break;

                case (true, false):
                    Compute.MatrixMultiply(
                        b.Value.Data, tensor.Gradient.Data, gradA,
                        b0, b1, a1, b1,
                        transposeA: false, transposeB: true // transpose grad
                    ); // d/da a^T x b = b x L'(O)^T, m = a0, k = b1, n = a1
                    // b is b0xb1, grad is a1xb1 (transposed is b1xa1)
                    
                    Compute.MatrixMultiply(
                        a.Value.Data, tensor.Gradient.Data, gradB,
                        a0, a1, a1, b1,
                        transposeA: false, transposeB: false // transpose none
                    ); // d/db a^T x b = a x L'(O), m = b0, k = a1, n = b1
                    // a is a0xa1, grad is a1xb1
                    break;

                case (false, true):
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, b.Value.Data, gradA,
                        a0, b0, b0, b1,
                        transposeA: false, transposeB: false // transpose none
                    ); // d/da a x b^T = L'(O) x b, m = a0, k = b0, n = a1
                    // grad is a0xb0, b is b0xb1

                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, a.Value.Data, gradB,
                        a0, b0, a0, a1,
                        transposeA: true, transposeB: false // transpose grad
                    ); // d/db a x b^T = L'(O)^T x a, m = b0, k = a0, n = b1
                    // grad is a0xb0 (transposed is b0xa0), a is a0xa1
                    break;

                case (true, true):
                    Compute.MatrixMultiply(
                        b.Value.Data, tensor.Gradient.Data, gradA,
                        b0, b1, a1, b0,
                        transposeA: true, transposeB: true // transpose both b and grad
                    ); // d/da a^T x b^T = b^T x L'(O)^T, m = a0, k = b0, n = a1
                    // b is b0xb1 (transposed is b1xb0), grad is a1xb0 (transposed is b0xa1)
    
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data, a.Value.Data, gradB,
                        a1, b0, a0, a1,
                        transposeA: true, transposeB: true // transpose both grad and a
                    ); // d/db a^T x b^T = L'(O)^T x a, m = b0, k = a1, n = b1
                    // grad is a1xb0 (transposed is b0xa1), a is a0xa1 (transposed is a1xa0)
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