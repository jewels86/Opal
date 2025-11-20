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
    
    public void UpdateWith(ITensorStorage<T> newValue);
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
    
    public void UpdateWith(ITensorStorage<T> newValue) => Data = newValue.ToHost();
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
