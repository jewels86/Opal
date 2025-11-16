using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    #region Scalar Tensor Helpers
    public static ITensorStorage<double> NewCpuScalarStorage(double value) => 
        new CpuStorage<double>(value, [1], 1);
    
    public static ITensorStorage<double> NewGpuScalarStorage(double value)
    {
        var buffer = Operations.Accelerator.Allocate1D<double>(1);
        buffer.CopyFromCPU([value]);
        return new GpuScalarStorage(buffer);
    }
    public static ITensorStorage<double> NewDefaultScalarStorage(double value) => NewCpuScalarStorage(value);
    public static ScalarTensor NewScalar(ITensorStorage<double> storage, List<object>? inputs, Action<Tensor<ITensorStorage<double>>> backwards,
        ITensorStorage<double> gradient) => new(storage, inputs, backwards, gradient);
    public static ScalarTensor NewScalar(double value, double gradient) => NewScalar(NewDefaultScalarStorage(value), null, _ => { }, NewDefaultScalarStorage(gradient));
    #endregion
    #region Storage Operations
    public static ITensorStorage<double> SubtractStorage(ITensorStorage<double> a, ITensorStorage<double> b) => NewCpuScalarStorage(a.ToHost() - b.ToHost());
    public static ITensorStorage<double> MultiplyStorage(ITensorStorage<double> a, ITensorStorage<double> b) => NewCpuScalarStorage(a.ToHost() * b.ToHost());
    public static ITensorStorage<double> AddStorage(ITensorStorage<double> a, ITensorStorage<double> b) => NewCpuScalarStorage(a.ToHost() + b.ToHost());
    #endregion
    public static ScalarTensor Add(ScalarTensor a, ScalarTensor b)
    {
        var result = a.Value.ToHost() + b.Value.ToHost();
        return new ScalarTensor(
            NewCpuScalarStorage(result),
            [a, b],
            Backward,
            NewCpuScalarStorage(0.0));
    
        void Backward(Tensor<ITensorStorage<double>> output)
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
}