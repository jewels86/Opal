using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    public static GpuScalarStorage ToGpuScalar(ScalarTensorStorage storage) => (storage as GpuScalarStorage) ?? (GpuScalarStorage)storage.ToGpu();
    public static MemoryBuffer1D<double, Stride1D.Dense> AllocateScalar() => AllocateBuffer(1);
    
    #region Scalar Tensor Helpers
    public static ScalarTensorStorage NewCpuScalarStorage(double value) => 
        new CpuStorage<double>(value, [1], 1);
    
    public static ScalarTensorStorage NewGpuScalarStorage(double value)
    {
        var buffer = Operations.Accelerator.Allocate1D<double>(1);
        buffer.CopyFromCPU([value]);
        return new GpuScalarStorage(buffer);
    }
    public static ScalarTensorStorage NewDefaultScalarStorage(double value) => NewCpuScalarStorage(value);
    public static ScalarTensor NewScalar(ScalarTensorStorage storage, List<object>? inputs, Action<Tensor<ScalarTensorStorage>> backwards,
        ScalarTensorStorage gradient) => new(storage, inputs, backwards, gradient);
    public static ScalarTensor NewScalar(double value, double gradient) => NewScalar(NewDefaultScalarStorage(value), null, _ => { }, NewDefaultScalarStorage(gradient));
    #endregion
    #region Storage Operations
    public static ScalarTensorStorage SubtractStorage(ScalarTensorStorage a, ScalarTensorStorage b) => NewCpuScalarStorage(a.ToHost() - b.ToHost());
    public static ScalarTensorStorage MultiplyStorage(ScalarTensorStorage a, ScalarTensorStorage b) => NewCpuScalarStorage(a.ToHost() * b.ToHost());
    public static ScalarTensorStorage AddStorage(ScalarTensorStorage a, ScalarTensorStorage b) => NewCpuScalarStorage(a.ToHost() + b.ToHost());
    #endregion
    #region Operations
    public static ScalarTensor Add(ScalarTensor a, ScalarTensor b)
    {
        var result = a.Value.ToHost() + b.Value.ToHost();
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [a, b],
            Backward,
            NewCpuScalarStorage(0.0));
    
        void Backward(ScalarTensor output)
        {
            var outGrad = output.Gradient.ToHost();
            a.Gradient.CopyFrom(a.Gradient.ToHost() + outGrad);
            b.Gradient.CopyFrom(b.Gradient.ToHost() + outGrad);
        }
    }
    public static ScalarTensor Multiply(ScalarTensor a, ScalarTensor b)
    {
        var aHost = a.Value.ToHost();
        var bHost = b.Value.ToHost();
        var result = aHost * bHost;
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [a, b],
            Backward,
            NewCpuScalarStorage(0.0));
    
        void Backward(ScalarTensor output)
        {
            var outGrad = output.Gradient.ToHost();
            a.Gradient.CopyFrom(a.Gradient.ToHost() + outGrad * bHost);
            b.Gradient.CopyFrom(b.Gradient.ToHost() + outGrad * aHost);;
        }
    }
    public static ScalarTensor Subtract(ScalarTensor a, ScalarTensor b)
    {
        var result = a.Value.ToHost() - b.Value.ToHost();
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [a, b],
            Backward,
            NewCpuScalarStorage(0.0));
    
        void Backward(ScalarTensor output)
        {
            var outGrad = output.Gradient.ToHost();
            a.Gradient.CopyFrom(a.Gradient.ToHost() - outGrad);
            b.Gradient.CopyFrom(b.Gradient.ToHost() - outGrad);
        }
    }
    #endregion
}