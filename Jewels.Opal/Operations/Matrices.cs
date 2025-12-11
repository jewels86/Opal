using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

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
    public static Tensor<float[,]> New(Value<float[,]> matrix, Value<float[,]>? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new(matrix, gradient ?? matrix.Zeros(), backwardAction, inputs);

    public static Value<float[,]> NewValue(float[,] matrix) => new MatrixValue(matrix, DefaultAcceleratorIndex);
    
    #region Kernels
    /// <summary>
    /// (result, matrix, vector, n) => result = matrix + vector[i % n]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int>[]
        AddVectorToMatrixKernel { get; } = Compute.Load((Index1D i,
            ArrayView1D<float, Stride1D.Dense> result,
            ArrayView1D<float, Stride1D.Dense> matrix,
            ArrayView1D<float, Stride1D.Dense> vector,
            int n) => result[i] = matrix[i] + vector[i % n]);

    /// <summary>
    /// (vectorGrad, grad, cols, rows) => vectorGrad[col] += sum(grad through rows)
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int, int>[] AddVectorToMatrixBackwardKernel { get; }
        = Compute.Load((Index1D col,
            ArrayView1D<float, Stride1D.Dense> vectorGrad,
            ArrayView1D<float, Stride1D.Dense> grad, int cols, int rows) =>
        {
            float sum = 0;
            for (int row = 0; row < rows; row++) sum += grad[row * cols + col];
            vectorGrad[col] += sum;
        });
    
    /// <summary>
    /// (result, a, b, colsA, colsB) => result[i] = a[row, col] if col less than colsA else b[row, colsB + (col - colsA)]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int>[]
        ConcatMatricesKernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        int colsA, int colsB) => 
        {
            int totalCols = colsA + colsB;
            var (row, col) = (i / totalCols, i % totalCols);
            if (col < colsA) result[i] = a[row * colsA + col];
            else result[i] = b[row * colsB + (col - colsA)];
        });
    
    /// <summary>
    /// (gradA, gradB, gradConcat, colsA, colsB) => gradA[row, col] = gradConcat[i] if col less than colsA else gradB[row, colsB + (col - colsA)]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int>[]
        ConcatMatricesBackwardKernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> gradA,
        ArrayView1D<float, Stride1D.Dense> gradB,
        ArrayView1D<float, Stride1D.Dense> gradConcat,
        int colsA, int colsB) => 
        {
            int totalCols = colsA + colsB;
            var (row, col) = (i / totalCols, i % totalCols);
            if (col < colsA) gradA[row * colsA + col] = gradConcat[i];
            else gradB[row * colsB + (col - colsA)] = gradConcat[i];
        });
    #endregion
    
    #region Multiplication
    public static Tensor<float[,]> MatrixMultiply(Tensor<float[,]> a, Tensor<float[,]> b, bool transposeA = false, bool transposeB = false)
    {
        var (aidx, a0, a1, b0, b1) = (a.AcceleratorIndex, a.Value.Shape[0], a.Value.Shape[1], b.Value.Shape[0], b.Value.Shape[1]);
        var m = transposeA ? a1 : a0;
        var n = transposeB ? b0 : b1;
        var kA = transposeA ? a0 : a1;
        var kB = transposeB ? b1 : b0;

        if (kA != kB) throw new Exception($"Matrices of shapes {ToString([m, kA])} and {ToString([kB, n])} (after transposing if necessary- flags a={transposeA}, b={transposeB}) cannot be multiplied.");
        
        var result = new MatrixValue(Compute.Get(aidx, m * n), [m, n]);
        Compute.MatrixMultiply(result, a.Value, b.Value, a0, a1, b0, b1, transposeA: transposeA, transposeB: transposeB);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var gradA = Compute.Get(aidx, a0 * a1);
            var gradB = Compute.Get(aidx, b0 * b1);

            switch (transposeA, transposeB)
            {
                case (false, false):
                    Compute.MatrixMultiply(
                        gradA, tensor.Gradient.Data, b.Value.Data, a0, b1, b0, b1,
                        transposeA: false, transposeB: true // transpose b
                    ); // d/da axb = L'(O) x b^T, m = a0, k = b1, n = a1
                    // grad is a0xb1, b is b0xb1 (transposed is b1xb0)
    
                    Compute.MatrixMultiply(
                        gradB, a.Value.Data, tensor.Gradient.Data, 
                        a0, a1, a0, b1,
                        transposeA: true, transposeB: false // transpose a
                    ); // d/db axb = a^T x L'(O), m = b0, k = a0, n = b1
                    // a is a0xa1 (transposed is a1xa0), grad is a0xb1
                    break;

                case (true, false):
                    Compute.MatrixMultiply(
                        gradA, b.Value.Data, tensor.Gradient.Data, 
                        b0, b1, a1, b1,
                        transposeA: false, transposeB: true // transpose grad
                    ); // d/da a^T x b = b x L'(O)^T, m = a0, k = b1, n = a1
                    // b is b0xb1, grad is a1xb1 (transposed is b1xa1)
                    
                    Compute.MatrixMultiply(
                        gradB, a.Value.Data, tensor.Gradient.Data, 
                        a0, a1, a1, b1,
                        transposeA: false, transposeB: false // transpose none
                    ); // d/db a^T x b = a x L'(O), m = b0, k = a1, n = b1
                    // a is a0xa1, grad is a1xb1
                    break;

                case (false, true):
                    Compute.MatrixMultiply(
                        gradA, tensor.Gradient.Data, b.Value.Data, 
                        a0, b0, b0, b1,
                        transposeA: false, transposeB: false // transpose none
                    ); // d/da a x b^T = L'(O) x b, m = a0, k = b0, n = a1
                    // grad is a0xb0, b is b0xb1

                    Compute.MatrixMultiply(
                        gradB, tensor.Gradient.Data, a.Value.Data, 
                        a0, b0, a0, a1,
                        transposeA: true, transposeB: false // transpose grad
                    ); // d/db a x b^T = L'(O)^T x a, m = b0, k = a0, n = b1
                    // grad is a0xb0 (transposed is b0xa0), a is a0xa1
                    break;

                case (true, true):
                    Compute.MatrixMultiply(
                        gradA, b.Value.Data, tensor.Gradient.Data, 
                        b0, b1, a1, b0,
                        transposeA: true, transposeB: true // transpose both b and grad
                    ); // d/da a^T x b^T = b^T x L'(O)^T, m = a0, k = b0, n = a1
                    // b is b0xb1 (transposed is b1xb0), grad is a1xb0 (transposed is b0xa1)
    
                    Compute.MatrixMultiply(
                        gradB, tensor.Gradient.Data, a.Value.Data, 
                        a1, b0, a0, a1,
                        transposeA: true, transposeB: true // transpose both grad and a
                    ); // d/db a^T x b^T = L'(O)^T x a, m = b0, k = a1, n = b1
                    // grad is a1xb0 (transposed is b0xa1), a is a0xa1 (transposed is a1xa0)
                    break;
            }

            Compute.Call(AccumulateKernels, a.Gradient, gradA);
            Compute.Call(AccumulateKernels, b.Gradient, gradB);
            Compute.Return(gradA, gradB);
        }
    }

    public static Tensor<float[]> MatrixVectorMultiply(Tensor<float[,]> matrix, Tensor<float[]> vector)
    {
        var (aidx, m, n) = (matrix.AcceleratorIndex, matrix.Value.Shape[0], matrix.Value.Shape[1]);
        var result = new VectorValue(Compute.Get(aidx, m));
        Compute.MatrixVectorMultiply(result, matrix.Value, vector.Value, m, n);
        return new(result, result.Zeros(), Backward, [matrix, vector]);

        void Backward(ITensor tensor)
        {
            var gradMatrix = Compute.Get(aidx, m * n);
            Compute.OuterProduct(gradMatrix, tensor.Gradient.Data, vector.Value.Data, m, n);
            Compute.Call(AccumulateKernels, matrix.Gradient, gradMatrix);
            
            var gradVector = Compute.Get(aidx, n);
            Compute.MatrixVectorMultiply(gradVector, matrix.Value.Data, tensor.Gradient.Data, m, n, transposeMatrix: true);
            Compute.Call(AccumulateKernels, vector.Gradient, gradVector);
            
            Compute.Return(gradMatrix, gradVector);
        }
    }
    #endregion
    
    public static Tensor<float[,]> Add(Tensor<float[,]> matrix, Tensor<float[]> vector)
    {
        var (aidx, rows, cols) = (matrix.AcceleratorIndex, matrix.Value.Shape[0], matrix.Value.Shape[1]);
        var result = new MatrixValue(Compute.Get(aidx, rows * cols), [rows, cols]);
        Compute.Call(AddVectorToMatrixKernel, result.Data, matrix.Value.Data, vector.Value.Data, cols);
    
        return new(result, result.Zeros(), Backward, [matrix, vector]);

        void Backward(ITensor tensor)
        {
            Compute.Call(AccumulateKernels, matrix.Gradient, tensor.Gradient.Data);
            Compute.Call(AddVectorToMatrixBackwardKernel, vector.Gradient.Data, tensor.Gradient.Data, cols, rows);
        }
    }

    public static Tensor<float[,]> Concat(Tensor<float[,]> a, Tensor<float[,]> b)
    {
        var (aidx, rows, colsA, colsB) = (a.AcceleratorIndex, a.Value.Shape[0], a.Value.Shape[1], b.Value.Shape[1]);
        if (a.Value.Shape[0] != b.Value.Shape[0]) throw new ArgumentException($"Concatenating matrices with different dimensions along axis- {ToString(a.Value.Shape)} vs {ToString(b.Value.Shape)} along d0");
        var result = new MatrixValue(Compute.Get(aidx, rows * (colsA + colsB)), [rows, colsA + colsB]);
        Compute.Call(ConcatMatricesKernel, result, a.Value.Data, b.Value.Data,  colsA, colsB);

        return new(result, result.Zeros(), Backward, [a, b]);
        
        void Backward(ITensor t)
        {
            var slicedA = Compute.Get(aidx, rows * colsA);
            var slicedB = Compute.Get(aidx, rows * colsB);
        
            Compute.Call(ConcatMatricesBackwardKernel,  slicedA, slicedB, t.Gradient.Data, colsA, colsB);
        
            Compute.Call(AccumulateKernels, a.Gradient, slicedA);
            Compute.Call(AccumulateKernels, b.Gradient, slicedB);
        }
    }
}