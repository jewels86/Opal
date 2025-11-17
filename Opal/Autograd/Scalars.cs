using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    public static GpuScalarStorage ToGpuScalar(ScalarTensorStorage storage) => (storage as GpuScalarStorage) ?? (GpuScalarStorage)storage.ToGpu();
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateScalar() => AllocateBuffer(1);
    public static bool UseGpu(params ScalarTensorStorage[] storages) => GpuAvailable && storages.Any(s => s is GpuScalarStorage);
    public static VectorTensorStorage VectorFromScalarStorage(ScalarTensorStorage storage) =>
        Operations.UseGpu(storage)
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
    public static ScalarTensor NewScalar(ScalarTensorStorage storage, List<object>? inputs, Action<ScalarTensor> backwards,
        ScalarTensorStorage gradient) => new(storage, inputs, backwards, gradient);
    public static ScalarTensor NewScalar(double value, double gradient) => NewScalar(NewDefaultScalarStorage(value), null, _ => { }, NewDefaultScalarStorage(gradient));
    #endregion
    #region Storage Operations
    public static void AccumulateGradient(ScalarTensorStorage gradient, ScalarTensorStorage incomingGrad)
    {
        if (!UseGpu(gradient, incomingGrad)) gradient.CopyFrom(incomingGrad.ToHost() + gradient.ToHost());
        else
        {
            var gpuGrad = ToGpuScalar(gradient);
            var gpuIncoming = ToGpuScalar(incomingGrad);
            Queue.Enqueue(() => VectorAddKernel(Index1D.One, gpuGrad.GpuData.View, gpuIncoming.GpuData.View, gpuGrad.GpuData.View));
        }
    }

    public static ScalarTensorStorage SubtractStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() - b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        Queue.Enqueue(() => VectorSubtractKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View));
        return new GpuScalarStorage(result);;
    }
    public static ScalarTensorStorage MultiplyStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() * b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        Queue.Enqueue(() => VectorMultiplyKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View));
        return new GpuScalarStorage(result);;
    }
    public static ScalarTensorStorage AddStorage(ScalarTensorStorage a, ScalarTensorStorage b)
    {
        if(!UseGpu(a, b)) return NewCpuScalarStorage(a.ToHost() + b.ToHost());
        var gpuA = ToGpuScalar(a);
        var gpuB = ToGpuScalar(b);
        var result = AllocateScalar();
        
        Queue.Enqueue(() => VectorAddKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View));
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
        Queue.Enqueue(() => {
            VectorAddKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        });
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
            Queue.Enqueue(() => VectorMultiplyKernel(
                Index1D.One, 
                gpuA.GpuData.View, 
                gpuB.GpuData.View, 
                result.View));
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
        Queue.Enqueue(() => {
            VectorSubtractKernel(Index1D.One, gpuA.GpuData.View, gpuB.GpuData.View, result.View);
        });
        return new ScalarTensor(new GpuScalarStorage(result), [a, b], Backward, NewGpuScalarStorage(0.0));

        void Backward(ScalarTensor output)
        {
            AccumulateGradient(a.Gradient, output.Gradient);
            AccumulateGradient(b.Gradient, NegateStorage(output.Gradient));
        }
    }
    
    public static ScalarTensor Negate(ScalarTensor a) => Multiply(a, NewScalar(-1.0, 0.0));
    #endregion
}