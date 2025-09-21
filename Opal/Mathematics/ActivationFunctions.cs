using static Opal.Mathematics.Matrices;

namespace Opal.Mathematics;

public static class ActivationFunctions
{
    #region Scalars
    public static readonly ActivationFunction<double> ReLu = new(
        x => Math.Max(0, x),
        x => x > 0 ? 1 : 0
    );

    public static readonly ActivationFunction<double> Sigmoid = new(
        x => 1.0 / (1.0 + Math.Exp(-x)),
        x => {
            var s = 1.0 / (1.0 + Math.Exp(-x));
            return s * (1 - s);
        }
    );

    public static readonly ActivationFunction<double> Tanh = new(
        Math.Tanh,
        x => 1 - Math.Pow(Math.Tanh(x), 2)
    );
    
    public static readonly ActivationFunction<double> Identity = new(
        x => x,
        x => 1.0
    );
    #endregion
    #region Vectors
    public static readonly ActivationFunction<double[]> ReLuVector = new(
        x => x.Select(ReLu.Function).ToArray(),
        x => x.Select(ReLu.Derivative).ToArray()
    );

    public static readonly ActivationFunction<double[]> SigmoidVector = new(
        x => x.Select(Sigmoid.Function).ToArray(),
        x => x.Select(Sigmoid.Derivative).ToArray()
    );

    public static readonly ActivationFunction<double[]> TanhVector = new(
        x => x.Select(Tanh.Function).ToArray(),
        x => x.Select(Tanh.Derivative).ToArray()
    );
    #endregion
    #region Matrices
    public static readonly ActivationFunction<double[,]> ReLuMatrix = new(
        x => ApplyElementwise(x, ReLu.Function),
        x => ApplyElementwise(x, ReLu.Derivative)
    );

    public static readonly ActivationFunction<double[,]> SigmoidMatrix = new(
        x => ApplyElementwise(x, Sigmoid.Function),
        x => ApplyElementwise(x, Sigmoid.Derivative)
    );

    public static readonly ActivationFunction<double[,]> TanhMatrix = new(
        x => ApplyElementwise(x, Tanh.Function),
        x => ApplyElementwise(x, Tanh.Derivative)
    );
    #endregion
}

public record struct ActivationFunction<T>(Func<T, T> Function, Func<T, T> Derivative);