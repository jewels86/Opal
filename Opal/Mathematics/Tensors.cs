namespace Opal.Mathematics;

public static class Tensors
{
    readonly private static Random Random = new();
    
    public static double RandomDouble(double min = -1, double max = 1)
    {
        return Random.NextDouble() * (max - min) + min;
    }
    
    #region Miscellaneous
    public static double Softmax(double[] values, int index)
    {
        double max = values.Max();
        double sumExp = values.Select(v => Math.Exp(v - max)).Sum();
        return Math.Exp(values[index] - max) / sumExp;
    }
    public static double[] Softmax(double[] values)
    {
        double max = values.Max();
        double sumExp = values.Select(v => Math.Exp(v - max)).Sum();
        return values.Select(v => Math.Exp(v - max) / sumExp).ToArray();
    }
    #endregion
}