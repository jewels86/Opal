using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd;

public class GpuKernels
{
    public static void AddKernel(
        Index1D index, 
        ArrayView1D<double, Stride1D.Dense> a, 
        ArrayView1D<double, Stride1D.Dense> b,
        ArrayView1D<double, Stride1D.Dense> result) =>
        result[index] = a[index] + b[index];
}