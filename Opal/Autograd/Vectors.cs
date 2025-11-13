using System.Runtime.CompilerServices;
using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    public static Tensor<double[]> Sum(params List<Tensor<double[]>> vectors)
    {
        var result = Vectors.Add(vectors.Select(v => v.Value).ToList());
        return new(result, vectors.Cast<object>().ToList(), Backwards, Vectors.Ones(result.Length));
        
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
}