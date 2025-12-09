namespace Jewels.Opal.Light;

public class Scalar(float value, float? gradient = null, List<Scalar>? inputs = null, Action<Scalar>? backwardAction = null)
{
    public float Value { get; set; } = value;
    public float Gradient { get; set; } = gradient.GetValueOrDefault(0);
    public List<Scalar> Inputs { get; set; } = inputs ?? [];
    public Action<Scalar> BackwardAction { get; set; } = backwardAction ?? (_ => { });
    
    #region Backward
    public void Backward(float initialGradient)
    {
        (List<Scalar> topo, HashSet<Scalar> visited) = ([], []);
        Build(this, topo, visited);

        Gradient = initialGradient;
        foreach (var node in topo.AsEnumerable().Reverse()) node.BackwardAction(node);
    }

    private static void Build(Scalar node, List<Scalar> topo, HashSet<Scalar> visited)
    {
        if (!visited.Add(node)) return;
        foreach (var input in node.Inputs) Build(input, topo, visited);
        topo.Add(node);
    }
    #endregion
    
    public static Scalar operator +(Scalar a, Scalar b) => ScalarOperations.Add(a, b);
    public static Scalar operator -(Scalar a, Scalar b) => ScalarOperations.Subtract(a, b);
    public static Scalar operator *(Scalar a, Scalar b) => ScalarOperations.Multiply(a, b);
    public static Scalar operator /(Scalar a, Scalar b) => ScalarOperations.Divide(a, b);
    public static Scalar operator -(Scalar a) => ScalarOperations.Negate(a);
    
    public static Scalar Square(Scalar a) => ScalarOperations.Square(a);
    public static Scalar Sqrt(Scalar a) => ScalarOperations.Sqrt(a);
    public static Scalar Sine(Scalar a) => ScalarOperations.Sine(a);
    public static Scalar Cosine(Scalar a) => ScalarOperations.Cosine(a);
    public static Scalar Tangent(Scalar a) => ScalarOperations.Tangent(a);
}

internal static class ScalarOperations
{
    public static Scalar Add(Scalar a, Scalar b) => new(a.Value + b.Value, 0f, [a, b], s =>
    {
        a.Gradient += s.Gradient;
        b.Gradient += s.Gradient;
    });

    public static Scalar Subtract(Scalar a, Scalar b) => new(a.Value - b.Value, 0f, [a, b], s =>
    {
        a.Gradient += s.Gradient;
        b.Gradient -= s.Gradient;
    });

    public static Scalar Multiply(Scalar a, Scalar b) => new(a.Value * b.Value, 0f, [a, b], s =>
    {
        a.Gradient += s.Gradient * b.Value;
        b.Gradient += s.Gradient * a.Value;
    });
    
    public static Scalar Divide(Scalar a, Scalar b) => new(a.Value / b.Value, 0f, [a, b], s =>
    {
        a.Gradient += s.Gradient / b.Value;
        b.Gradient -= s.Gradient * a.Value / (b.Value * b.Value);
    });
    
    public static Scalar Negate(Scalar a) => new(-a.Value, 0f, [a], s => a.Gradient -= s.Gradient);
    
    public static Scalar Square(Scalar a) => Multiply(a, a);
    public static Scalar Sqrt(Scalar a) => 
        new(MathF.Sqrt(a.Value), 0f, [a], s => a.Gradient += s.Gradient / (2f * MathF.Sqrt(a.Value)));
    
    public static Scalar Sine(Scalar a) => 
        new(MathF.Sin(a.Value), 0f, [a], s => a.Gradient += s.Gradient * MathF.Cos(a.Value));
    public static Scalar Cosine(Scalar a) => 
        new(MathF.Cos(a.Value), 0f, [a], s => a.Gradient -= s.Gradient * MathF.Sin(a.Value));
    public static Scalar Tangent(Scalar a) => 
        new(MathF.Tan(a.Value), 0f, [a], s => a.Gradient += s.Gradient * (1f + (MathF.Tan(a.Value) * MathF.Tan(a.Value))));
}