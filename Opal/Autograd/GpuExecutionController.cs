using System.Diagnostics;
using System.Diagnostics.Tracing;
using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd;

public class GpuExecutionController
{
    public bool AutoExecute { get; set; } = false;
    
    private readonly Accelerator _accelerator;
    private readonly Dictionary<int, Stack<MemoryBuffer1D<double, Stride1D.Dense>>> _vectorPools = [];
    private readonly Dictionary<(int, int), Stack<MemoryBuffer2D<double, Stride2D.DenseX>>> _matrixPools = [];
    private readonly List<MemoryBuffer1D<double, Stride1D.Dense>> _vectorDeferred = [];
    private readonly List<MemoryBuffer2D<double, Stride2D.DenseX>> _matrixDeferred = [];

    public GpuExecutionController(Accelerator accelerator) => _accelerator = accelerator;
    public int Count = 0;

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

    public MemoryBuffer1D<double, Stride1D.Dense> GetTemp(int size)
    {
        var buffer = Get(size);
        DeferReturn(buffer);
        return buffer;
    }

    public MemoryBuffer2D<double, Stride2D.DenseX> GetTemp(int rows, int cols)
    {
        var buffer = Get(rows, cols);
        DeferReturn(buffer);
        return buffer;
    }

    public void Return(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        //Console.WriteLine($"Return {buffer.Length}");
        int size = (int)buffer.Length;
        if (!_vectorPools.TryGetValue(size, out var pool))
        {
            pool = new();
            _vectorPools[size] = pool;
        }

        //if (pool.Contains(buffer)) return;
        
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

        pool.Push(buffer);
    }
    public void DeferReturn(MemoryBuffer1D<double, Stride1D.Dense> buffer)
    {
        _vectorDeferred.Add(buffer);
    }

    public void DeferReturn(MemoryBuffer2D<double, Stride2D.DenseX> buffer) => _matrixDeferred.Add(buffer);
    public void DeferReturn(params MemoryBuffer1D<double, Stride1D.Dense>[] buffers) => _vectorDeferred.AddRange(buffers);
    public void DeferReturn(params MemoryBuffer2D<double, Stride2D.DenseX>[] buffers) => _matrixDeferred.AddRange(buffers);

    public void Return(params MemoryBuffer1D<double, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    }
    public void Return(params MemoryBuffer2D<double, Stride2D.DenseX>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    }
    
    public void Sync()
    {
        //Console.WriteLine($"Sync!");
        _accelerator.Synchronize();
        foreach (var buffer in _vectorDeferred) Return(buffer);
        foreach (var buffer in _matrixDeferred) Return(buffer);
        _vectorDeferred.Clear();
        _matrixDeferred.Clear();
    }

    public void Flush()
    {
        foreach (var pool in _vectorPools.Values)
        {
            foreach (var buffer in pool)
                buffer.Dispose();
        }
        foreach (var pool in _matrixPools.Values)
        {
            foreach (var buffer in pool)
                buffer.Dispose();
        }
        _vectorPools.Clear();
        _matrixPools.Clear();
    }

    private MemoryBuffer1D<double, Stride1D.Dense> TryGetFrom(Stack<MemoryBuffer1D<double, Stride1D.Dense>> pool, int size)
    {
        //Console.WriteLine($"Getting from {size} (pool count: {pool.Count})");
        if (pool.Count > 0) return pool.Pop();
        
        var buffer = _accelerator.Allocate1D<double>(size);
        Operations.VectorFillKernel(buffer.IntExtent, buffer.View, 0.0);
        return buffer;
    }

    private MemoryBuffer2D<double, Stride2D.DenseX> TryGetFrom(Stack<MemoryBuffer2D<double, Stride2D.DenseX>> pool, int rows, int cols)
    {
        if (pool.Count > 0) return pool.Pop();
        
        var buffer = _accelerator.Allocate2DDenseX<double>(new LongIndex2D(rows, cols));
        Operations.MatrixFillKernel(buffer.IntExtent, buffer.View, 0.0);
        return buffer;
    }
}