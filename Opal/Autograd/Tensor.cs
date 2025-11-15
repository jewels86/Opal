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
    public required T Data { get; set; }
    public required int[] Shape { get; set; }
    public required int TotalElements { get; set; }
    
    public T ToHost() => Data;
    public void CopyFrom(T source) => Data = source;
}

public class GpuVectorStorage : ITensorStorage<double[]>
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
}

public class GpuMatrixStorage : ITensorStorage<double[,]>
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
        
        AddKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>>(GpuKernels.AddKernel);
    }
    
    public static Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>, 
        ArrayView1D<double, Stride1D.Dense>> AddKernel { get; private set; }

    public static VectorTensor BinaryOp(
        VectorTensor a, 
        VectorTensor b,
        Action<Index1D, ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>, 
            ArrayView1D<double, Stride1D.Dense>> gpuKernel,
        Func<double[], double[], double[]> cpuFallback,
        Action<VectorTensor, VectorTensor, VectorTensor> gradientFn)
    {
        if (a.Value is GpuVectorStorage gpuA && b.Value is GpuVectorStorage gpuB)
        {
            var resultBuffer = Accelerator.Allocate1D<double>(gpuA.GpuData.Length);
        
            Queue.Enqueue(() => gpuKernel((int)gpuA.GpuData.Length, gpuA.GpuData.View, gpuB.GpuData.View, resultBuffer.View));
        
            var resultStorage = new GpuVectorStorage(resultBuffer);
            var gradStorage = new GpuVectorStorage(Accelerator.Allocate1D<double>((int)resultBuffer.Length));
        
            return new VectorTensor(resultStorage, [a, b], 
                output => gradientFn(a, b, (VectorTensor)output), gradStorage);
        }
        var result = cpuFallback(a.Value.ToHost(), b.Value.ToHost());
        return new VectorTensor(
            VectorTensor.CpuVectorStorage(result),
            [a, b],
            output => gradientFn(a, b, (VectorTensor)output),
            VectorTensor.CpuVectorStorage(new double[result.Length]));
    }
    
    public static void AccumulateGradient(
        ITensorStorage<double[]> gradient,
        ITensorStorage<double[]> incomingGrad)
    {
        if (gradient is GpuVectorStorage gpuGrad && incomingGrad is GpuVectorStorage gpuIncoming)
        {
            AddKernel(
                (int)gpuGrad.GpuData.Length,
                gpuGrad.GpuData.View,
                gpuIncoming.GpuData.View,
                gpuGrad.GpuData.View);
            Accelerator.Synchronize();
        }
        else
        {
            var gradData = gradient.ToHost();
            var incomingData = incomingGrad.ToHost();
            gradient.CopyFrom(Vectors.Add(gradData, incomingData));
        }
    }
    
    public static void Sync() => Queue.Execute();
}
