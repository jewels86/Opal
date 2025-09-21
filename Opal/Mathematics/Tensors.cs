namespace Opal.Mathematics;

public static class Tensors
{
    private static readonly Random Random = new();
    
    public static double RandomDouble(double min = -1, double max = 1)
    {
        return Random.NextDouble() * (max - min) + min;
    }
}