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
}