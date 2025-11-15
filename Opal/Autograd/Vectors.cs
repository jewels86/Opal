using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using Opal.Mathematics;

namespace Opal.Autograd;

public class VectorTensor : Tensor<ITensorStorage<double[]>>
{
    public VectorTensor(
        ITensorStorage<double[]> storage, 
        List<object>? inputs, 
        Action<Tensor<ITensorStorage<double[]>>>? backward,
        ITensorStorage<double[]> gradient)
        : base(storage, inputs, backward ?? (_ => { }), gradient) {}

    public static ITensorStorage<double[]> CpuVectorStorage(double[] vector) => 
        new CpuStorage<double[]> { Data = vector, Shape = [vector.Length], TotalElements = vector.Length };

    public static ITensorStorage<double[]> GpuVectorStorage(double[] vector)
    {
        var buffer = Operations.Accelerator.Allocate1D<double>(vector.Length);
        buffer.CopyFromCPU(vector);
        return new GpuVectorStorage(buffer);
    }
}

public static partial class Operations
{
    public static VectorTensor Add(VectorTensor a, VectorTensor b) => 
        BinaryOp(
            a, b, 
            AddKernel, Vectors.Add, 
            (_, _, output) =>
            {
                AccumulateGradient(a.Gradient, output.Gradient);
                AccumulateGradient(b.Gradient, output.Gradient);
            });
    
    public static VectorTensor Sum(params List<VectorTensor> vectors)
    {
        var result = Vectors.Add(vectors.Select(v => v.Value).ToList());
        return new(result, vectors.Cast<object>().ToList(), Backwards, Vectors.Zeros(result.Length));
        
        void Backwards(Tensor<double[]> output)
        {
            foreach (var vector in vectors) vector.Gradient = Vectors.Add(vector.Gradient, output.Gradient);
        }
    }

    public static Tensor<double[]> Multiply(params List<Tensor<double[]>> vectors)
    {
        var result = Vectors.Multiply(vectors.Select(v => v.Value).ToList());
        return new(result, vectors.Cast<object>().ToList(), Backwards, Vectors.Zeros(result.Length));
        
        void Backwards(Tensor<double[]> output)
        {
            foreach (var vector in vectors)
            {
                var productOfOthers = Vectors.Multiply(vectors
                    .Where(v => v != vector)
                    .Select(v => v.Value)
                    .ToList());
        
                vector.Gradient = Vectors.Add(vector.Gradient, Vectors.Multiply(output.Gradient, productOfOthers));
            }
        }
    }

    public static Tensor<double[]> Subtract(Tensor<double[]> vector, Tensor<double[]> other)
    {
        var result = Vectors.Subtract(vector.Value, other.Value);
        return new(result, new List<object> {vector, other}, Backwards, Vectors.Zeros(result.Length));
        
        void Backwards(Tensor<double[]> output)
        {
            vector.Gradient = Vectors.Add(vector.Gradient, output.Gradient);
            other.Gradient = Vectors.Subtract(other.Gradient, output.Gradient);
        }
    }
    
    public static Tensor<double> Dot(Tensor<double[]> a, Tensor<double[]> b)
    {
        var result = Vectors.Dot(a.Value, b.Value);
        List<object> inputs = [a, b];
    
        return new Tensor<double>(result, inputs, Backwards, 0.0);
    
        void Backwards(Tensor<double> output)
        {
            a.Gradient = Vectors.Add(a.Gradient, Vectors.Multiply(b.Value, output.Gradient));
        
            b.Gradient = Vectors.Add(b.Gradient, Vectors.Multiply(a.Value, output.Gradient));
        }
    }

    public static Tensor<double[]> Concat(Tensor<double[]> a, Tensor<double[]> b)
    {
        var result = Vectors.Concat(a.Value, b.Value);
        List<object> inputs = [a, b];
    
        return new Tensor<double[]>(result, inputs, Backwards, Vectors.Zeros(result.Length));
    
        void Backwards(Tensor<double[]> output)
        {
            var aLength = a.Value.Length;
        
            var gradA = output.Gradient[..aLength];
            var gradB = output.Gradient[aLength..];
        
            a.Gradient = Vectors.Add(a.Gradient, gradA);
            b.Gradient = Vectors.Add(b.Gradient, gradB);
        }
    }
}