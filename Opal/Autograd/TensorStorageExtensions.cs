namespace Opal.Autograd;

public static class TensorStorageExtensions
{
    public static ScalarTensorStorage ToGpu(this ScalarTensorStorage storage)
    {
        if (storage is GpuScalarStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuScalarStorage(data);
    }
    public static VectorTensorStorage ToGpu(this VectorTensorStorage storage)
    {
        if (storage is GpuVectorStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuVectorStorage(data);
    }
    
    public static MatrixTensorStorage ToGpu(this MatrixTensorStorage storage)
    {
        if (storage is GpuMatrixStorage)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewGpuMatrixStorage(data);
    }

    public static ScalarTensorStorage ToCpu(this ScalarTensorStorage storage)
    {
        if (storage is CpuStorage<double>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuScalarStorage(data);
    }
    
    public static VectorTensorStorage ToCpu(this VectorTensorStorage storage)
    {
        if (storage is CpuStorage<double[]>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuVectorStorage(data);
    }
    
    public static MatrixTensorStorage ToCpu(this MatrixTensorStorage storage)
    {
        if (storage is CpuStorage<double[,]>)
            return storage;
        
        var data = storage.ToHost();
        return Operations.NewCpuMatrixStorage(data);
    }
    
    public static TStorage Defer<TStorage>(this TStorage storage) 
        where TStorage : IDisposable
    {
        storage.Dispose();
        return storage;
    }

    public static TStorage Replace<TStorage>(this TStorage storage, TStorage replacement)
        where TStorage : IDisposable
    {
        storage.Dispose();
        return replacement;
    }

    public static void UpdateWith(this ScalarTensorStorage storage, ScalarTensorStorage newValue)
    {
        if (storage is GpuScalarStorage gpuStorage && newValue is GpuScalarStorage gpuNewValue)
            Operations.VectorCopyKernel(
                gpuStorage.GpuData.IntExtent,
                gpuNewValue.GpuData.View,
                gpuStorage.GpuData.View);
        else
            storage.CopyFrom(newValue.ToHost());
        newValue.Dispose();
    }
    public static void UpdateWith(this VectorTensorStorage storage, VectorTensorStorage newValue)
    {
        if (storage is GpuVectorStorage gpuStorage && newValue is GpuVectorStorage gpuNewValue)
        {
            Operations.VectorCopyKernel(
                gpuStorage.GpuData.IntExtent, 
                gpuNewValue.GpuData.View, 
                gpuStorage.GpuData.View);
        }
        else
            storage.CopyFrom(newValue.ToHost());
        newValue.Dispose();
    }
    public static void UpdateWith(this MatrixTensorStorage storage, MatrixTensorStorage newValue)
    {
        if (storage is GpuMatrixStorage gpuStorage && newValue is GpuMatrixStorage gpuNewValue)
        {
            Operations.MatrixCopyKernel(
                gpuStorage.GpuData.IntExtent, 
                gpuNewValue.GpuData.View, 
                gpuStorage.GpuData.View);
        }
        else
            storage.CopyFrom(newValue.ToHost());
        newValue.Dispose();
    }
}