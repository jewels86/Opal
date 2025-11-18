using System.Diagnostics;
using System.Diagnostics.Tracing;
using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd;

public class GpuExecutionQueue
{
    public bool AutoExecute { get; set; } = false;
    
    private readonly Accelerator _accelerator;
    private readonly Queue<Action> _operations = [];
    private readonly Dictionary<int, Stack<MemoryBuffer1D<double, Stride1D.Dense>>> _vectorPools = [];
    private readonly Dictionary<(int, int), Stack<MemoryBuffer2D<double, Stride2D.DenseX>>> _matrixPools = [];
    
    public int Threshold { get; set; } = 50;
    
    public GpuExecutionQueue(Accelerator accelerator) => _accelerator = accelerator;

    public MemoryBuffer1D<double, Stride1D.Dense> Get(int size)
    {
        if (_vectorPools.TryGetValue(size, out var pool)) return TryGetFrom(pool, size);
        pool = new();
        _vectorPools[size] = pool;
        return TryGetFrom(pool, size);
    }

    public MemoryBuffer2D<double, Stride2D.DenseX> Get(int rows, int cols)
    {
        if (_matrixPools.TryGetValue((rows, cols), out var pool)) return TryGetFrom(pool, rows, cols);
        pool = new();
        _matrixPools[(rows, cols)] = pool;
        
        return TryGetFrom(pool, rows, cols);
    }

    public void Return(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        int size = (int)buffer.Length;
        if (!_vectorPools.TryGetValue(size, out var pool))
        {
            pool = new();
            _vectorPools[size] = pool;
        }

        if (pool.Contains(buffer)) return;

        Operations.VectorFillKernel(buffer.IntExtent, buffer.View, 0.0);
        _accelerator.Synchronize();

        pool.Push(buffer);
    }
    public void Return(MemoryBuffer2D<double, Stride2D.DenseX> buffer)
    {
        (int rows, int cols) = ((int)buffer.Extent.X, (int)buffer.Extent.Y);
        if (!_matrixPools.TryGetValue((rows, cols), out var pool))
        {
            pool = new();
            _matrixPools[(rows, cols)] = pool;
        }

        if (pool.Contains(buffer)) return;

        Operations.MatrixFillKernel(buffer.IntExtent, buffer.View, 0.0);
        _accelerator.Synchronize();

        pool.Push(buffer);
    }

    public void Enqueue(Action operation)
    {
        if (AutoExecute)
        {
            operation();
            return;
        }
        
        _operations.Enqueue(operation);
        if (_operations.Count >= Threshold) Execute();
    }

    public void Execute()
    {
        if (_operations.Count == 0) return;
        //Console.WriteLine($"Executing {_operations.Count} operations");
        
        while (_operations.Count > 0) _operations.Dequeue()();
        _accelerator.Synchronize();
        _operations.Clear();
    }
    
    public void Flush()
    {
        Execute();

        foreach (var pool in _vectorPools.Values)
        {
            while (pool.Count > 0) pool.Pop().Dispose();
        }
        _vectorPools.Clear();
        
        foreach (var pool in _matrixPools.Values)
        {
            while (pool.Count > 0) pool.Pop().Dispose();
        }
        _matrixPools.Clear();
    }

    private MemoryBuffer1D<double, Stride1D.Dense> TryGetFrom(Stack<MemoryBuffer1D<double, Stride1D.Dense>> pool, int size)
    {
        if (pool.Count > 0) return pool.Pop();
        
        var buffer = _accelerator.Allocate1D<double>(size);
        Operations.VectorFillKernel(buffer.IntExtent, buffer.View, 0.0);
        _accelerator.Synchronize();
        return buffer;
    }

    private MemoryBuffer2D<double, Stride2D.DenseX> TryGetFrom(Stack<MemoryBuffer2D<double, Stride2D.DenseX>> pool, int rows, int cols)
    {
        if (pool.Count > 0) return pool.Pop();
        
        var buffer = _accelerator.Allocate2DDenseX<double>(new LongIndex2D(rows, cols));
        Operations.MatrixFillKernel(buffer.IntExtent, buffer.View, 0.0);
        _accelerator.Synchronize();
        return buffer;
    }
}