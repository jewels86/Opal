namespace Opal.Autograd;

public static class TensorStorageExtensions
{
    public static ITensorStorage<double> ToGpu(this ITensorStorage<double> storage)
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

    public static ITensorStorage<double> ToCpu(this ITensorStorage<double> storage)
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
}