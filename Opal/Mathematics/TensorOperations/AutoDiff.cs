namespace Opal.Mathematics.TensorOperations;

public class Tracked<T>
{
    public T Value { get; set; }
    public int NodeID { get; set; }
    
    public static implicit operator T(Tracked<T> tracked) => tracked.Value;
    public static implicit operator Tracked<T>(T value) => new() {Value = value};
    
}

public class ADNode<T>
{
    public required int ID { get; init; }
    public required Action<Tracked<T>> Backwards { get; init; }
}