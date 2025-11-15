using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd;

using ScalarTensor = Tensor<ITensorStorage<double>>;
using VectorTensor = Tensor<ITensorStorage<double[]>>;
using MatrixTensor = Tensor<ITensorStorage<double[,]>>;

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

    public double[] ToHost() => GpuData.GetAsArray1D();
    public void CopyFrom(double[] data) => GpuData.CopyFromCPU(data);
}

public class GpuMatrixStorage : ITensorStorage<double[,]>
{
    public MemoryBuffer2D<double, Stride2D.DenseX> GpuData { get; set; }
    public int[] Shape => [(int)GpuData.Extent.X, (int)GpuData.Extent.Y];
    public int TotalElements => (int)GpuData.Length;
    
    public GpuMatrixStorage(MemoryBuffer2D<double, Stride2D.DenseX> gpuData) => GpuData = gpuData;
    
    public double[,] ToHost() => GpuData.GetAsArray2D();
    public void CopyFrom(double[,] data) => GpuData.CopyFromCPU(data);
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