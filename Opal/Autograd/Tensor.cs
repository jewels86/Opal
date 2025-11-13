namespace Opal.Autograd;

public class Tensor<T> where T : notnull
{
    public T Value { get; set; }
    public List<object>? Inputs { get; set; }
    public Action<Tensor<T>> Backwards { get; set; }
    public T Gradient { get; set; }

    public Tensor(T value, List<object>? inputs, Action<Tensor<T>> backwards, T gradient) =>
        (Value, Inputs, Backwards, Gradient) = (value, inputs, backwards, gradient);
    
    public void Backward(T initialGradient)
    {
        var topo = new List<object>();
        var visited = new HashSet<object>();
        BuildTopo(this, topo, visited);

        Gradient = initialGradient;
        
        foreach (var node in topo.AsEnumerable().Reverse()) ((dynamic)node).Backwards((dynamic)node);
    }
    
    private static void BuildTopo(object node, List<object> topo, HashSet<object> visited)
    {
        if (!visited.Add(node)) return;

        if (((dynamic)node).Inputs is List<object> inputs)
        {
            foreach (var input in inputs)
                BuildTopo(input, topo, visited);
        }

        topo.Add(node);
    }
}