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
        new CpuStorage<double> 
        { 
            Data = value, 
            Shape = Array.Empty<int>(), 
            TotalElements = 1 
        };
    
    public static ITensorStorage<double> GpuScalarStorage(double value)
    {
        var buffer = Operations.Accelerator.Allocate1D<double>(1);
        buffer.CopyFromCPU([value]);
        return new GpuScalarStorage(buffer);
    }
}

public static partial class Operations
{
    public static Tensor<double> Sum(params List<Tensor<double>> scalars)
    {
        var result = scalars.Sum(s => s.Value);
        return new(result, scalars.Cast<object>().ToList(), Backwards, 0.0);

        void Backwards(Tensor<double> output)
        {
            foreach (var scalar in scalars) scalar.Gradient += output.Gradient;
        }
    }

    public static Tensor<double> Multiply(params List<Tensor<double>> scalars)
    {
        var result = scalars.Aggregate(1.0, (a, s) => a * s.Value);

        return new(result, scalars.Cast<object>().ToList(), Backwards, 0.0);

        void Backwards(Tensor<double> output)
        {
            foreach (var scalar in scalars)
            {
                var productOfOthers = scalars
                    .Where(s => s != scalar)
                    .Aggregate(1.0, (acc, s) => acc * s.Value);

                scalar.Gradient += output.Gradient * productOfOthers;
            }
        }
    }

    public static Tensor<double> Subtract(Tensor<double> scalar, Tensor<double> other)
    {
        var result = scalar.Value - other.Value;
        return new(result, new List<object> {scalar, other}, Backwards, 0.0);
        
        void Backwards(Tensor<double> output)
        {
            scalar.Gradient += output.Gradient;
            other.Gradient -= output.Gradient;
        }
    }
    
    public static Tensor<double[]> VectorFromScalars(params Tensor<double>[] scalars)
    {
        var result = scalars.Select(s => s.Value).ToArray();
        List<object> inputs = scalars.Cast<object>().ToList();
    
        return new Tensor<double[]>(result, inputs, Backwards, Vectors.Zeros(result.Length));
    
        void Backwards(Tensor<double[]> output)
        {
            for (int i = 0; i < scalars.Length; i++)
                scalars[i].Gradient += output.Gradient[i];
        }
    }
    
    public static Tensor<double> ZeroGrad(double value) => new(value, null, _ => { }, 0.0);
}