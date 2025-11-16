using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd;

public static class GpuKernels
{
    #region Vectors
    public static void VectorAddKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] + b[index];
    
    public static void VectorMultiplyKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] * b[index];
    
    public static void ScalarVectorMultiplyKernel(Index1D index, 
        ArrayView1D<double, Stride1D.Dense> scalar, 
        ArrayView1D<double, Stride1D.Dense> vector,
        ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = scalar[0] * vector[index];
    
    public static void VectorConcatKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> a,
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result,
        int aLength)
    {
        if (index < aLength)
            result[index] = a[index];
        else
            result[index] = b[index - aLength];
    }

    public static void VectorSliceKernel(
        Index1D index,
        ArrayView1D<double, Stride1D.Dense> source,
        ArrayView1D<double, Stride1D.Dense> dest,
        int offset) =>
        dest[index] = source[index + offset];
    public static void VectorNegateKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> vector, 
        ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = -vector[index];
    #endregion
    #region Matrices
    public static void MatrixAddKernel(
        Index2D index,
        ArrayView2D<double, Stride2D.DenseX> a,
        ArrayView2D<double, Stride2D.DenseX> b,
        ArrayView2D<double, Stride2D.DenseX> result) =>
        result[index] = a[index] + b[index];
    
    public static void MatrixMultiplyKernel(Index2D index,
        ArrayView2D<double, Stride2D.DenseX> a,
        ArrayView2D<double, Stride2D.DenseX> b,
        ArrayView2D<double, Stride2D.DenseX> result) =>
        result[index] = a[index] * b[index];
    
    public static void MatrixVectorMultiplyKernel(
        Index1D row,
        ArrayView2D<double, Stride2D.DenseX> matrix,
        ArrayView1D<double, Stride1D.Dense> vector,
        ArrayView1D<double, Stride1D.Dense> result)
    {
        double sum = 0;
        for (int col = 0; col < matrix.Extent.X; col++)
        {
            sum += matrix[col, row] * vector[col];
        }
        result[row] = sum;
    }
    public static void MatrixTransposeVectorMultiplyKernel(
        Index1D col,
        ArrayView2D<double, Stride2D.DenseX> matrix,
        ArrayView1D<double, Stride1D.Dense> vector,
        ArrayView1D<double, Stride1D.Dense> result)
    {
        double sum = 0;
        for (int row = 0; row < matrix.Extent.Y; row++)
        {
            sum += matrix[col, row] * vector[row];
        }
        result[col] = sum;
    }
    public static void OuterProductKernel(
        Index2D index,
        ArrayView1D<double, Stride1D.Dense> a,
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView2D<double, Stride2D.DenseX> result) =>
        result[index] = a[index.Y] * b[index.X];

    public static void CopyVectorToRowKernel(
        Index1D col,
        ArrayView1D<double, Stride1D.Dense> vector,
        ArrayView2D<double, Stride2D.DenseX> matrix,
        int row) =>
        matrix[col, row] = vector[col];

    public static void ScaleVectorByRowKernel(
        Index1D col,
        ArrayView1D<double, Stride1D.Dense> vector,
        ArrayView1D<double, Stride1D.Dense> scalars,
        ArrayView1D<double, Stride1D.Dense> result,
        int scalarIndex) =>
        result[col] = vector[col] * scalars[scalarIndex];
    #endregion
}