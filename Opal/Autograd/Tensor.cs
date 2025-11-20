using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using Opal.Mathematics;
using static System.GC;

namespace Opal.Autograd;

public interface ITensorStorage<T> : IDisposable where T : notnull
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
    public void Dispose() { }
}

public class GpuScalarStorage : ScalarTensorStorage
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
    
    public void Dispose()
    {
        Operations.Controller.DeferReturn(GpuData);
        GC.SuppressFinalize(this);
    }

    public static implicit operator MemoryBuffer1D<double, Stride1D.Dense>(GpuScalarStorage storage) => storage.GpuData;
    public static implicit operator ArrayView1D<double, Stride1D.Dense>(GpuScalarStorage storage) => storage.GpuData.View;
}

public class GpuVectorStorage : VectorTensorStorage
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
    
    public void Dispose()
    {
        Operations.Controller.DeferReturn(GpuData);
        GC.SuppressFinalize(this);
    }

    public static implicit operator MemoryBuffer1D<double, Stride1D.Dense>(GpuVectorStorage storage) => storage.GpuData;
    public static implicit operator ArrayView1D<double, Stride1D.Dense>(GpuVectorStorage storage) => storage.GpuData.View;
}

public class GpuMatrixStorage : MatrixTensorStorage
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
    
    public void Dispose()
    {
        Operations.Controller.DeferReturn(GpuData);
        SuppressFinalize(this);
    }
    
    public static implicit operator MemoryBuffer2D<double, Stride2D.DenseX>(GpuMatrixStorage storage) => storage.GpuData;
    public static implicit operator ArrayView2D<double, Stride2D.DenseX>(GpuMatrixStorage storage) => storage.GpuData.View;
}

public interface ITensor : IDisposable 
{
    List<ITensor>? Inputs { get; }
}

public class Tensor<T> : ITensor where T : notnull
{
    public T Value { get; set; }
    public List<ITensor>? Inputs { get; set; }
    public Action<Tensor<T>> Backwards { get; set; }
    public T Gradient { get; set; }
    
    private bool _disposed;
    private readonly object _lock = new();

    public Tensor(T value, List<ITensor>? inputs, Action<Tensor<T>> backwards, T gradient) => (Value, Inputs, Backwards, Gradient) = (value, inputs, backwards, gradient);

    public void Backward(T initialGradient)
    {
        var topo = new List<ITensor>();
        var visited = new HashSet<ITensor>();
        BuildTopo(this, topo, visited);

        Gradient = initialGradient;
        
        foreach (var node in topo.AsEnumerable().Reverse()) ((dynamic)node).Backwards((dynamic)node);
    }
    
    private static void BuildTopo(ITensor node, List<ITensor> topo, HashSet<ITensor> visited)
    {
        if (!visited.Add(node)) return;

        if (node.Inputs != null)
            foreach (var input in node.Inputs)
                BuildTopo(input, topo, visited);

        topo.Add(node);
    }

    

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            
            DisposeValues();

            if (Inputs is null) return;
            foreach (var input in Inputs) input.Dispose();
        }
    }

    public void MarkDisposed() => _disposed = true;

    public void DisposeValues()
    {
        (Value as IDisposable)?.Dispose();
        (Gradient as IDisposable)?.Dispose();
    }
    public static implicit operator T(Tensor<T> tensor) => tensor.Value;
}
