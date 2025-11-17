using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace Opal.Autograd;

public static partial class Operations
{
    public static Context Context { get; private set; }
    public static Accelerator Accelerator { get; private set; }
    public static GpuExecutionQueue Queue { get; } 
    public static bool GpuAvailable { get; set; }

    static Operations()
    {
        Context = Context.Create(builder => builder
            .Default()
            .EnableAlgorithms());
        
        try
        {
            Accelerator = Context.CreateCudaAccelerator(0);
            GpuAvailable = true;
        }
        catch
        {
            Accelerator = Context.CreateCPUAccelerator(0);
            GpuAvailable = false;
        }
        Queue = new(Accelerator);
        
        VectorAddKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorAddKernel);
        VectorMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorMultiplyKernel);
        VectorDivideKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorDivideKernel);
        ScalarVectorMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.ScalarVectorMultiplyKernel);
        VectorConcatKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int>(GpuKernels.VectorConcatKernel);
        VectorSliceKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, int>(GpuKernels.VectorSliceKernel);
        VectorNegateKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorNegateKernel);
        VectorSubtractKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorSubtractKernel);
        VectorFillKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, double>(GpuKernels.VectorFillKernel);
        VectorFillScalarKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorFillScalarKernel);
        VectorPowerKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorPowerKernel);
        VectorLogKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorLogKernel);
        VectorSqrtKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorSqrtKernel);
        VectorMaxKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorMaxKernel);
        VectorTanhKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorTanhKernel);
        VectorExpKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorExpKernel);
        VectorScalarMaxKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.VectorScalarMaxKernel);
        
        MatrixVectorMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.MatrixVectorMultiplyKernel);
        MatrixTransposeVectorMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<double, Stride2D.DenseX>,
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.MatrixTransposeVectorMultiplyKernel);
        OuterProductKernel = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>>(GpuKernels.OuterProductKernel);
        MatrixAddKernel = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(GpuKernels.MatrixAddKernel);
        CopyVectorToRowKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
            ArrayView2D<double, Stride2D.DenseX>, int>(GpuKernels.CopyVectorToRowKernel);
        ScaleVectorByRowKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int>(GpuKernels.ScaleVectorByRowKernel);
        MatrixSubtractKernel = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(GpuKernels.MatrixSubtractKernel);
        MatrixScalarMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<double, Stride2D.DenseX>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView2D<double, Stride2D.DenseX>>(GpuKernels.MatrixScalarMultiplyKernel);
    }
    
    public static void Sync() => Queue.Execute();
}