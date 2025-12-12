using Jewels.Lazulite;

namespace Jewels.Opal;


public interface ITensor : IDisposable 
{
    public List<ITensor> Inputs { get; }
    public void Backward(IValue initialGradient);
    public Action<ITensor> BackwardAction { get; }
    public IValue Value { get; }
    public IValue Gradient { get; }
    
    public bool Disposable { get; set; }
    public int AcceleratorIndex { get; }
    public int[] Shape { get; }
    public int TotalSize { get; }

    public ITensor Create(IValue value, IValue gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null);
}

public class Tensor<T>(Value<T> value, Value<T>? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) : ITensor
    where T : notnull
{
    public Value<T> Value { get; } = value;
    public Value<T> Gradient { get; } = gradient ?? value.Zeros();
    public Action<ITensor> BackwardAction { get; set; } = backwardAction ?? (_ => { });
    public List<ITensor> Inputs { get; } = inputs ?? [];

    public bool Disposable
    {
        get => !Value.Disposable && !Gradient.Disposable;
        set
        {
            Value.Disposable = value;
            Gradient.Disposable = value;
        }
    }

    public int AcceleratorIndex => Value.AcceleratorIndex;
    public int[] Shape => Value.Shape;
    public int TotalSize => Value.TotalSize;
    
    IValue ITensor.Value => Value;
    IValue ITensor.Gradient => Gradient;
    public ITensor Create(IValue value, IValue gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new Tensor<T>((Value<T>)value, (Value<T>)gradient, backwardAction, inputs);
    public Tensor<T> Create(Value<T> value, Value<T> gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) =>
        new(value, gradient, backwardAction, inputs);

    private bool _isDisposed;

    #region Backward Pass
    public void Backward(IValue initialGradient)
    {
        (List<ITensor> topo, HashSet<ITensor> visited) = ([], []);
        Build(this, topo, visited);

        Gradient.UpdateWith((Value<T>)initialGradient); // initial gradient should be of the same type as the tensor's gradient- if not we would throw anyway
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
    
    public static implicit operator Value<T>(Tensor<T> tensor) => tensor.Value;
    
    public static Tensor<T> operator +(Tensor<T> a, Tensor<T> b) => Operations.Add(a, b);
    public static Tensor<T> operator *(Tensor<T> a, Tensor<T> b) => Operations.Multiply(a, b);
    public static Tensor<T> operator -(Tensor<T> a, Tensor<T> b) => Operations.Subtract(a, b);
    public static Tensor<T> operator /(Tensor<T> a, Tensor<T> b) => Operations.Divide(a, b);
    public static Tensor<T> operator -(Tensor<T> a) => Operations.Negate(a);
}

public static class TensorExtensions
{
    public static Tensor<T> Defer<T>(this Tensor<T> tensor) where T : notnull
    {
        tensor.Dispose();
        return tensor;
    }

    public static ITensor Defer(this ITensor tensor)
    {
        tensor.Dispose();
        return tensor;
    }
    
    public static Tensor<T> Disposable<T>(this Tensor<T> tensor) where T : notnull
    {
        tensor.Disposable = true;
        return tensor;
    }
    
    public static Tensor<T> NonDisposable<T>(this Tensor<T> tensor) where T : notnull
    {
        tensor.Disposable = false;
        return tensor;
    }
    
    public static ITensor Disposable(this ITensor tensor)
    {
        tensor.Disposable = true;
        return tensor;
    }
    
    public static ITensor NonDisposable(this ITensor tensor)
    {
        tensor.Disposable = false;
        return tensor;
    }
}