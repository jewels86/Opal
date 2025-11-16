using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using Opal.Mathematics;


namespace Opal.Autograd;

public interface ITensorStorage<T> where T : notnull
{
    public T ToHost();
    public void CopyFrom(T source);
    public int[] Shape { get; }
    public int TotalElements { get; }
}

public class CpuStorage<T> : ITensorStorage<T> where T : notnull
{
    public CpuStorage(T data, int[] shape, int totalElements)
    {
        Data = data;
        Shape = shape;
        TotalElements = totalElements;
    }

    public T Data { get; set; }
    public int[] Shape { get; set; }
    public int TotalElements { get; set; }
    
    public T ToHost() => Data;
    public void CopyFrom(T source) => Data = source;
}

public class GpuScalarStorage : ITensorStorage<double>, IDisposable
{
    public MemoryBuffer1D<double, Stride1D.Dense> GpuData { get; set; }
    public int[] Shape => [1];
    public int TotalElements => 1;
    
    public GpuScalarStorage(MemoryBuffer1D<double, Stride1D.Dense> gpuData) => GpuData = gpuData;

    public double ToHost()
    {
        Operations.Sync();
        return GpuData.GetAsArray1D()[0];
    }

    public void CopyFrom(double source)
    {
        Operations.Sync();
        GpuData.CopyFromCPU([source]);
    }

    public GpuVectorStorage ToVector() => new(GpuData);
    
    public void Dispose() => GpuData.Dispose();
}

public class GpuVectorStorage : ITensorStorage<double[]>, IDisposable
{
    public MemoryBuffer1D<double, Stride1D.Dense> GpuData { get; set; }
    public int[] Shape => [(int)GpuData.Length];
    public int TotalElements => (int)GpuData.Length;
    
    public GpuVectorStorage(MemoryBuffer1D<double, Stride1D.Dense> gpuData) => GpuData = gpuData;

    public double[] ToHost()
    {
        Operations.Sync();
        return GpuData.GetAsArray1D();
    }

    public void CopyFrom(double[] data)
    {
        Operations.Sync();
        GpuData.CopyFromCPU(data);
    }
    
    public void Dispose() => GpuData.Dispose();
}

public class GpuMatrixStorage : ITensorStorage<double[,]>, IDisposable
{
    public MemoryBuffer2D<double, Stride2D.DenseX> GpuData { get; set; }
    public int[] Shape => [(int)GpuData.Extent.X, (int)GpuData.Extent.Y];
    public int TotalElements => (int)GpuData.Length;
    
    public GpuMatrixStorage(MemoryBuffer2D<double, Stride2D.DenseX> gpuData) => GpuData = gpuData;

    public double[,] ToHost()
    {
        Operations.Sync();
        return GpuData.GetAsArray2D();
    }

    public void CopyFrom(double[,] data)
    {
        Operations.Sync();
        GpuData.CopyFromCPU(data);
    }
    
    public void Dispose() => GpuData.Dispose();
}

public class Tensor<T> where T : notnull
{
    public T Value { get; set; }
    public List<object>? Inputs { get; set; }
    public Action<Tensor<T>> Backwards { get; set; }
    public T Gradient { get; set; }

    public Tensor(T value, List<object>? inputs, Action<Tensor<T>> backwards, T gradient) =>
        (Value, Inputs, Backwards, Gradient) = (value, inputs, backwards, gradient);
    
    public void Backward(T initialGradient)
    {
        var topo = new List<object>();
        var visited = new HashSet<object>();
        BuildTopo(this, topo, visited);

        Gradient = initialGradient;
        
        foreach (var node in topo.AsEnumerable().Reverse()) ((dynamic)node).Backwards((dynamic)node);
    }
    
    private static void BuildTopo(object node, List<object> topo, HashSet<object> visited)
    {
        if (!visited.Add(node)) return;

        if (((dynamic)node).Inputs is List<object> inputs)
        {
            foreach (var input in inputs)
                BuildTopo(input, topo, visited);
        }

        topo.Add(node);
    }
    
    public void Dispose()
    {
        (Value as IDisposable)?.Dispose();
        (Gradient as IDisposable)?.Dispose();
    }
}

public static class TensorStorageExtensions
{
    public static ITensorStorage<double> ToGpu(this ITensorStorage<double> storage)
    {
        if (storage is GpuScalarStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuScalarStorage(data);
    }
    public static ITensorStorage<double[]> ToGpu(this ITensorStorage<double[]> storage)
    {
        if (storage is GpuVectorStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuVectorStorage(data);
    }
    
    public static ITensorStorage<double[,]> ToGpu(this ITensorStorage<double[,]> storage)
    {
        if (storage is GpuMatrixStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuMatrixStorage(data);
    }

    public static ITensorStorage<double> ToCpu(this ITensorStorage<double> storage)
    {
        if (storage is CpuStorage<double>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuScalarStorage(data);
    }
    
    public static ITensorStorage<double[]> ToCpu(this ITensorStorage<double[]> storage)
    {
        if (storage is CpuStorage<double[]>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuVectorStorage(data);
    }
    
    public static ITensorStorage<double[,]> ToCpu(this ITensorStorage<double[,]> storage)
    {
        if (storage is CpuStorage<double[,]>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuMatrixStorage(data);
    }
}

public static partial class Operations
{
    public static Context Context { get; private set; }
    public static Accelerator Accelerator { get; private set; }
    public static GpuExecutionQueue Queue { get; } 
    public static bool GpuAvailable { get; }

    static Operations()
    {
        Context = Context.CreateDefault();
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
