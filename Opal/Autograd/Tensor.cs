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
    
    public void Dispose() => Operations.Queue.Return(GpuData);
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
    
    public void Dispose() => Operations.Queue.Return(GpuData);
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
    
    public void Dispose() => Operations.Queue.Return(GpuData);
}

public interface ITensor : IDisposable 
{
    List<object>? Inputs { get; }
    void DisposeValues();
    public void MarkDisposed();
}

public class Tensor<T> : ITensor where T : notnull
{
    public T Value { get; set; }
    public List<object>? Inputs { get; set; }
    public Action<Tensor<T>> Backwards { get; set; }
    public T Gradient { get; set; }
    
    private bool _disposed;

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
        if (node is not ITensor tensor || !visited.Add(node)) return;

        if (tensor.Inputs != null)
            foreach (var input in tensor.Inputs)
                BuildTopo(input, topo, visited);

        topo.Add(node);
    }
    
    public void DisposeGraph()
    {
        var stack = new Stack<ITensor>();
        var disposed = new HashSet<ITensor>();
    
        stack.Push(this);
    
        while (stack.Count > 0)
        {
            var node = stack.Pop();
            if (!disposed.Add(node)) continue;
        
            node.DisposeValues();
            node.MarkDisposed();

            if (node.Inputs == null) continue;
            foreach (var input in node.Inputs)
                if (input is ITensor tensor)
                    stack.Push(tensor);
        }
    }
    
    public void DisposeValues()
    {
        (Value as IDisposable)?.Dispose();
        (Gradient as IDisposable)?.Dispose();
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        DisposeGraph();
        _disposed = true;
        SuppressFinalize(this);
    }
    
    public void MarkDisposed() => _disposed = true;
}
