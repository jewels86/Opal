using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    public static GpuScalarStorage ToGpuScalar(ScalarTensorStorage storage) => storage as GpuScalarStorage ?? (GpuScalarStorage)storage.ToGpu();
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateScalar() => AllocateBuffer(1);
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateTemp() => AllocateTemp(1);
    public static bool UseGpu(params ScalarTensorStorage[] storages) => GpuAvailable && storages.Any(s => s is GpuScalarStorage);
    public static VectorTensorStorage VectorFromScalarStorage(ScalarTensorStorage storage) =>
        UseGpu(storage)
            ? new GpuVectorStorage(Operations.ToGpuScalar(storage).GpuData)
            : new CpuStorage<double[]>([storage.ToHost()], [1], 1);
    
    #region Scalar Tensor Helpers
    public static ScalarTensorStorage NewCpuScalarStorage(double value) => 
        new CpuStorage<double>(value, [1], 1);
    
    public static ScalarTensorStorage NewGpuScalarStorage(double value)
    {
        var buffer = AllocateScalar();
        buffer.CopyFromCPU([value]);
        
        return new GpuScalarStorage(buffer);
    }
    public static ScalarTensorStorage NewDefaultScalarStorage(double value) => GpuAvailable ? NewGpuScalarStorage(value) : NewCpuScalarStorage(value);
    public static ScalarTensor NewScalar(ScalarTensorStorage storage, List<ITensor>? inputs, Action<ScalarTensor> backwards,
        ScalarTensorStorage gradient) => new(storage, inputs, backwards, gradient);
    public static ScalarTensor NewScalar(double value, double gradient) => NewScalar(NewDefaultScalarStorage(value), null, _ => { }, NewDefaultScalarStorage(gradient));
    #endregion
    #region Storage Operations
    public static void AccumulateGradient(ScalarTensorStorage gradient, ScalarTensorStorage incomingGrad)
    {
        if (!UseGpu(gradient, incomingGrad))
        {
            gradient.CopyFrom(gradient.ToHost() + incomingGrad.ToHost());
            return;
        }
    
        var gpuGrad = ToGpuScalar(gradient);
        var gpuIncoming = ToGpuScalar(incomingGrad);
        VectorAddKernel(Index1D.One, gpuGrad.GpuData.View, gpuIncoming.GpuData.View, gpuGrad.GpuData.View);

        if (gradient is GpuScalarStorage) return;
        var result = new double[1];
        gpuGrad.GpuData.CopyToCPU(result);
        gradient.CopyFrom(result[0]);
    }

    public static ScalarTensorStorage SubtractStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() - b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        VectorSubtractKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        return new GpuScalarStorage(result);;
    }
    public static ScalarTensorStorage MultiplyStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() * b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        VectorMultiplyKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        return new GpuScalarStorage(result);;
    }
    public static ScalarTensorStorage AddStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() + b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        VectorAddKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        return new GpuScalarStorage(result);
    }
    public static ScalarTensorStorage NegateStorage(ScalarTensorStorage a) => MultiplyStorage(a, NewDefaultScalarStorage(-1.0));
    #endregion
    #region Operations
    public static ScalarTensor Add(ScalarTensor a, ScalarTensor b)
    {
        if (!UseGpu(a.Value, b.Value))
        {
            return new ScalarTensor(
                NewCpuScalarStorage(a.Value.ToHost() + b.Value.ToHost()),
                [a, b],
                Backward,
                NewCpuScalarStorage(0.0));
        }
        var gpuA = ToGpuScalar(a.Value);
        var gpuB = ToGpuScalar(b.Value);
        var result = AllocateScalar();
        
        
        VectorAddKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        
        return new ScalarTensor(new GpuScalarStorage(result), [a, b], Backward, NewGpuScalarStorage(0.0));

        void Backward(ScalarTensor output)
        {
            AccumulateGradient(a.Gradient, output.Gradient);
            AccumulateGradient(b.Gradient, output.Gradient);
        }
    }
    public static ScalarTensor Multiply(ScalarTensor a, ScalarTensor b)
    {
        if (UseGpu(a.Value, b.Value))
        {
            var gpuA = ToGpuScalar(a.Value);
            var gpuB = ToGpuScalar(b.Value);
            var result = AllocateScalar();
            VectorMultiplyKernel(
                Index1D.One, 
                gpuA.GpuData.View, 
                gpuB.GpuData.View, 
                result.View);
            return new ScalarTensor(new GpuScalarStorage(result), [a, b], output =>
            {
                AccumulateGradient(a.Gradient, MultiplyStorage(output.Gradient, b.Value));
                AccumulateGradient(b.Gradient, MultiplyStorage(a.Value, output.Gradient));
            }, NewGpuScalarStorage(0.0));
        }
        var aHost = a.Value.ToHost();
        var bHost = b.Value.ToHost();
        return new ScalarTensor(
            NewCpuScalarStorage(aHost * bHost),
            [a, b],
            Backward,
            NewCpuScalarStorage(0.0));
    
        void Backward(ScalarTensor output)
        {
            var outGrad = output.Gradient.ToHost();
            a.Gradient.CopyFrom(a.Gradient.ToHost() + outGrad * bHost);
            b.Gradient.CopyFrom(b.Gradient.ToHost() + outGrad * aHost);
        }
    }
    
    public static ScalarTensor Subtract(ScalarTensor a, ScalarTensor b)
    {
        if (!UseGpu(a.Value, b.Value))
        {
            return new ScalarTensor(
                NewCpuScalarStorage(a.Value.ToHost() - b.Value.ToHost()),
                [a, b],
                Backward,
                NewCpuScalarStorage(0.0));
        }
        var gpuA = ToGpuScalar(a.Value);
        var gpuB = ToGpuScalar(b.Value);
        var result = AllocateScalar();
        
        VectorSubtractKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        
        return new ScalarTensor(new GpuScalarStorage(result), [a, b], Backward, NewGpuScalarStorage(0.0));

        void Backward(ScalarTensor output)
        {
            AccumulateGradient(a.Gradient, output.Gradient);
            AccumulateGradient(b.Gradient, NegateStorage(output.Gradient));
        }
    }
    
    public static ScalarTensor Negate(ScalarTensor a) => Multiply(a, NewScalar(-1.0, 0.0));
    #endregion
    
    public static ScalarTensorStorage One { get; }
    public static ScalarTensorStorage Zero { get; }
}