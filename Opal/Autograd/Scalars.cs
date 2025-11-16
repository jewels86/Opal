using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public class ScalarTensor : Tensor<ITensorStorage<double>>
{
    public ScalarTensor(
        ITensorStorage<double> storage, 
        List<object>? inputs, 
        Action<Tensor<ITensorStorage<double>>>? backward,
        ITensorStorage<double> gradient)
        : base(storage, inputs, backward ?? (_ => { }), gradient) {}
    
    public static ITensorStorage<double> CpuScalarStorage(double value) => 
        new CpuStorage<double>(value, [1], 1);
    
    public static ITensorStorage<double> GpuScalarStorage(double value)
    {
        var buffer = Operations.Accelerator.Allocate1D<double>(1);
        buffer.CopyFromCPU([value]);
        return new GpuScalarStorage(buffer);
    }
    
    public static ScalarTensor operator +(ScalarTensor a, ScalarTensor b) => Operations.Add(a, b);
    public static ScalarTensor operator -(ScalarTensor a, ScalarTensor b) => Operations.Subtract(a, b);
    public static ScalarTensor operator *(ScalarTensor a, ScalarTensor b) => Operations.Multiply(a, b);
}

public static partial class Operations
{
    public static ScalarTensor Add(ScalarTensor a, ScalarTensor b)
    {
        var result = a.Value.ToHost() + b.Value.ToHost();
        return new ScalarTensor(
            ScalarTensor.CpuScalarStorage(result),
            [a, b],
            Backward,
            ScalarTensor.CpuScalarStorage(0.0));
    
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
            ScalarTensor.CpuScalarStorage(result),
            [a, b],
            Backward,
            ScalarTensor.CpuScalarStorage(0.0));
    
        void Backward(Tensor<ITensorStorage<double>> output)
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
            ScalarTensor.CpuScalarStorage(result),
            [a, b],
            Backward,
            ScalarTensor.CpuScalarStorage(0.0));
    
        void Backward(Tensor<ITensorStorage<double>> output)
        {
            var outGrad = output.Gradient.ToHost();
            a.Gradient.CopyFrom(a.Gradient.ToHost() - outGrad);
            b.Gradient.CopyFrom(b.Gradient.ToHost() - outGrad);
        }
    }
}