using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd.Gpu;

public abstract class GpuHigherStorage<T> : ITensorStorage<T>
    where T : notnull
{
    public MemoryBuffer3D<double, Stride3D.DenseXY> GpuData { get; set; }
    public int[] Shape => [(int)GpuData.Extent.X, (int)GpuData.Extent.Y, (int)GpuData.Extent.Z];
    public int TotalElements => (int)GpuData.Length;
    
    public GpuHigherStorage(MemoryBuffer3D<double, Stride3D.DenseXY> gpuData) => GpuData = gpuData;

    public T ToHost()
    {
        Operations.Sync();
        return Unroll(GpuData.GetAsArray3D());
    }
    public void CopyFrom(T source)
    {
        Operations.Sync();
        GpuData.CopyFromCPU(Flatten(source));
    }

    public void UpdateWith(ITensorStorage<T> newValue)
    {
        //var gpuNewValue = Operations.ToGpuHigher(newValue);
        //Operations.HigherCopyKernel(
        //    gpuNewValue.GpuData.IntExtent, 
        //    gpuNewValue.GpuData.View, 
        //    GpuData.View);
    }

    public void Dispose()
    {
        //Operations.Controller.DeferReturn(GpuData);
        GC.SuppressFinalize(this);
    }

    protected abstract T Unroll(double[,,] flattened);
    protected abstract double[,,] Flatten(T host);
}