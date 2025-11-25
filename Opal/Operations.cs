using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static partial class Operations
{
    public static int DefaultAcceleratorIndex { get; }

    static Operations()
    {
        Compute.InitializeExtraKernels();
        DefaultAcceleratorIndex = Compute.RequestAccelerator();
    }

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> ElementwiseFloatMulAndSubKernels { get; }
        = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> r, float alpha) =>
            r[i] = b[i] - a[i] * alpha);

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseTripleAddKernels { get; } 
        = Compute.Load((i, a, b, c, r) => r[i] = a[i] + b[i] + c[i]);
}