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
    #endregion
}