using Opal.Autograd;
using static Opal.Mathematics.Matrices;

namespace Opal.Mathematics;

public static class ActivationFunctions
{
    #region Scalars
    public static readonly ActivationFunction<double> ReLu = new(
        x =>
        {
            var result = Math.Max(0, x.Value);
            return new Tensor<double>(result, [x], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                x.Gradient += x.Value > 0 ? output.Gradient : 0;
            }
        }
    );

    public static readonly ActivationFunction<double> Sigmoid = new(
        x =>
        {
            var s = 1.0 / (1.0 + Math.Exp(-x.Value));
            return new Tensor<double>(s, [x], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                var sig = output.Value;
                x.Gradient += output.Gradient * sig * (1 - sig);
            }
        }
    );

    public static readonly ActivationFunction<double> Tanh = new(
        x =>
        {
            var result = Math.Tanh(x.Value);
            return new Tensor<double>(result, [x], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                var t = Math.Tanh(x.Value);
                x.Gradient += output.Gradient * (1 - t * t);
            }
        }
    );
    
    public static readonly ActivationFunction<double> Identity = new(
        x =>
        {
            return new Tensor<double>(x.Value, [x], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                x.Gradient += output.Gradient;
            }
        }
    );
    #endregion
    #region Vectors
    public static readonly ActivationFunction<double[]> ReLuVector = new(
        x =>
        {
            var result = x.Value.Select(v => Math.Max(0, v)).ToArray();
            return new Tensor<double[]>(result, [x], Backwards, Vectors.Zeros(result.Length));
            
            void Backwards(Tensor<double[]> output)
            {
                var grad = x.Value.Zip(output.Gradient, (v, g) => v > 0 ? g : 0).ToArray();
                x.Gradient = Vectors.Add(x.Gradient, grad);
            }
        }
    );

    public static readonly ActivationFunction<double[]> SigmoidVector = new(
        x =>
        {
            var result = x.Value.Select(v => 1.0 / (1.0 + Math.Exp(-v))).ToArray();
            return new Tensor<double[]>(result, [x], Backwards, Vectors.Zeros(result.Length));
            
            void Backwards(Tensor<double[]> output)
            {
                var grad = x.Value.Zip(output.Gradient, (v, g) =>
                {
                    var s = 1.0 / (1.0 + Math.Exp(-v));
                    return g * s * (1 - s);
                }).ToArray();
                x.Gradient = Vectors.Add(x.Gradient, grad);
            }
        }
    );

    public static readonly ActivationFunction<double[]> TanhVector = new(
        x =>
        {
            var result = x.Value.Select(Math.Tanh).ToArray();
            return new Tensor<double[]>(result, [x], Backwards, Vectors.Zeros(result.Length));
            
            void Backwards(Tensor<double[]> output)
            {
                var grad = x.Value.Zip(output.Gradient, (v, g) =>
                {
                    var t = Math.Tanh(v);
                    return g * (1 - t * t);
                }).ToArray();
                x.Gradient = Vectors.Add(x.Gradient, grad);
            }
        }
    );
    #endregion
    #region Matrices
    public static readonly ActivationFunction<double[,]> ReLuMatrix = new(
        x =>
        {
            var result = ApplyElementwise(x.Value, v => Math.Max(0, v));
            return new Tensor<double[,]>(result, [x], Backwards, new double[result.GetLength(0), result.GetLength(1)]);
            
            void Backwards(Tensor<double[,]> output)
            {
                var grad = ApplyElementwise(x.Value, (v, i, j) => v > 0 ? output.Gradient[i, j] : 0);
                x.Gradient = Add(x.Gradient, grad);
            }
        }
    );

    public static readonly ActivationFunction<double[,]> SigmoidMatrix = new(
        x =>
        {
            var result = ApplyElementwise(x.Value, v => 1.0 / (1.0 + Math.Exp(-v)));
            return new Tensor<double[,]>(result, [x], Backwards, new double[result.GetLength(0), result.GetLength(1)]);
            
            void Backwards(Tensor<double[,]> output)
            {
                var grad = ApplyElementwise(x.Value, (v, i, j) =>
                {
                    var s = 1.0 / (1.0 + Math.Exp(-v));
                    return output.Gradient[i, j] * s * (1 - s);
                });
                x.Gradient = Add(x.Gradient, grad);
            }
        }
    );

    public static readonly ActivationFunction<double[,]> TanhMatrix = new(
        x =>
        {
            var result = ApplyElementwise(x.Value, Math.Tanh);
            return new Tensor<double[,]>(result, [x], Backwards, new double[result.GetLength(0), result.GetLength(1)]);
            
            void Backwards(Tensor<double[,]> output)
            {
                var grad = ApplyElementwise(x.Value, (v, i, j) =>
                {
                    var t = Math.Tanh(v);
                    return output.Gradient[i, j] * (1 - t * t);
                });
                x.Gradient = Add(x.Gradient, grad);
            }
        }
    );
    #endregion
}

public record struct ActivationFunction<T>(Func<Tensor<T>, Tensor<T>> Function) where T : notnull;
