using ILGPU;
using ILGPU.Runtime;

namespace Opal.Autograd.Gpu;

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
        GC.SuppressFinalize(this);
    }
    
    public static implicit operator MemoryBuffer2D<double, Stride2D.DenseX>(GpuMatrixStorage storage) => storage.GpuData;
    public static implicit operator ArrayView2D<double, Stride2D.DenseX>(GpuMatrixStorage storage) => storage.GpuData.View;
    
    public void UpdateWith(MatrixTensorStorage newValue)
    {
        var gpuNewValue = Operations.ToGpuMatrix(newValue);
        Operations.MatrixCopyKernel(
            GpuData.IntExtent, 
            gpuNewValue.GpuData.View, 
            GpuData.View);
    }
}