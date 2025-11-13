using Opal.Mathematics;

namespace Opal.Autograd;

public static partial class Operations
{
    public static Tensor<double> Sum(params List<Tensor<double>> scalars)
    {
        var result = scalars.Sum(s => s.Value);
        return new(result, scalars.Cast<object>().ToList(), Backwards, 1.0);

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
            scalar.Gradient -= output.Gradient;
            other.Gradient += output.Gradient;
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