namespace Opal.Utilities;

public static class MathFunctions
{
    public static double[] ReLu(double[] x) => x.Select(v => Math.Max(0, v)).ToArray();
    public static double[] ReLuDerivative(double[] x) => x.Select(v => v > 0 ? 1.0 : 0.0).ToArray();
}