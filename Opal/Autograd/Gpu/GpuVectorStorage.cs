using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd.Gpu;

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
    
    public void UpdateWith(VectorTensorStorage newValue)
    {
        var gpuNewValue = Operations.ToGpuVector(newValue);
        Operations.VectorCopyKernel(
            gpuNewValue.GpuData.IntExtent,
            gpuNewValue.GpuData.View,
            GpuData.View);
    }
}