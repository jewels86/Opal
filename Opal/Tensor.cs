using Jewels.Lazulite;

namespace Opal;


public interface ITensor : IDisposable 
{
    public List<ITensor> Inputs { get; }
    public void Backward(object initialGradient);
    public Action<ITensor> BackwardAction { get; set; }
    public IValue Value { get; }
    public IValue Gradient { get; }
}

public class Tensor<T>(Value<T> value, Value<T> gradient, Action<ITensor>? backwardFunction = null, List<ITensor>? inputs = null) : ITensor
    where T : notnull
{
    public Value<T> Value { get; set; } = value;
    public Value<T> Gradient { get; set; } = gradient;
    public Action<ITensor> BackwardAction { get; set; } = backwardFunction ?? (_ => { });
    public List<ITensor> Inputs { get; set; } = inputs ?? [];
    
    public int AcceleratorIndex => Value.AcceleratorIndex;
    
    IValue ITensor.Value => Value;
    IValue ITensor.Gradient => Gradient;
    
    private bool _isDisposed = false;

    #region Backward Pass
    // initialGradient should be of the same type as the tensor's value
    public void Backward(object initialGradient)
    {
        if (initialGradient is not Value<T> valueGrad) throw new ArgumentException("Invalid gradient type");
        (List<ITensor> topo, HashSet<ITensor> visited) = ([], []);
        Build(this, topo, visited);

        Gradient = valueGrad;
        foreach (var node in topo.AsEnumerable().Reverse()) node.BackwardAction(node);
    }

    private static void Build(ITensor node, List<ITensor> topo, HashSet<ITensor> visited)
    {
        if (!visited.Add(node)) return;
        foreach (var input in node.Inputs) Build(input, topo, visited);
        topo.Add(node);
    }
    #endregion

    public void Dispose()
    {
        if (_isDisposed) return;
        Value.Dispose();
        Gradient.Dispose();
        foreach (var input in Inputs) input.Dispose();
        _isDisposed = true;
    }
    
    ~Tensor() => Dispose();
    
    public static implicit operator Value<T>(Tensor<T> tensor) => tensor.Value;
}
