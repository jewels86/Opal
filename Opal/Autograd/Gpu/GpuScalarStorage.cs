using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd.Gpu;

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
    
    public void UpdateWith(ScalarTensorStorage newValue)
    {
        var gpuNewValue = Operations.ToGpuScalar(newValue);
        Operations.VectorCopyKernel(
            gpuNewValue.GpuData.IntExtent,
            gpuNewValue.GpuData.View,
            GpuData.View);
    }
}