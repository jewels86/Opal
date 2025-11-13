using Opal.NNs.Ff;

namespace Opal.Mathematics.TensorOperations;

public class ADScalarTensorOperations : IFfTensorOperations<double, double, double, double>
{
    public List<ADNode<double>> Tape { get; } = [];
    public Dictionary<int, double> Gradients { get; } = [];
    public int NextID { get; private set; }
    public Dictionary<double, int> ValueToNode { get;} = [];

    public bool Recording { get; private set; } = false;

    public void StartRecording()
    {
        Recording = true;
        Tape.Clear();
    }
    public void StopRecording() => Recording = false;
    public void ClearRecord() => Tape.Clear();
    public void ClearGradients() => Gradients.Clear();
    public void Clear() { ClearGradients(); ClearRecord(); }
    
    public void ZeroGradients() 
    {
        foreach (var key in Gradients.Keys.ToList()) Gradients[key] = 0;
    }

    #region Operations
    public double Add(double output, double biases) 
    {
        int outputId = GetOrCreateNodeId(output);
        int biasesId = GetOrCreateNodeId(biases);
        
        if (!Recording) return output + biases;
    
        return New(output + biases, (gradOutput) => {
            Gradients[outputId] += gradOutput.Value;
            Gradients[biasesId] += gradOutput.Value;
        });
    }

    public double Multiply(double weights, double input) 
    {
        int weightsId = GetOrCreateNodeId(weights);
        int inputId = GetOrCreateNodeId(input);
        double w = GetOrCreateNodeId(weights);
        double i = GetOrCreateNodeId(input);
        
        if (!Recording) return weights * input;
    
        return New(w * i, (gradOutput) => {
            Gradients[weightsId] += gradOutput.Value * i;
            Gradients[inputId] += gradOutput.Value * w;
        });
    }
    
    public
    #endregion
    
    public void Backward(double output, double gradOutput = 1.0) {
        int outputId = ValueToNode[output];
        Gradients[outputId] = gradOutput;
        
        for (int i = Tape.Count - 1; i >= 0; i--) {
            var node = Tape[i];
            if (!Gradients.TryGetValue(node.ID, out double gradient)) continue;
            var tracked = new Tracked<double> { 
                Value = gradient, 
                NodeID = node.ID 
            };
            node.Backwards(tracked);
        }
    }

    private int GetOrCreateNodeId(double value) {
        if (ValueToNode.TryGetValue(value, out int id)) return id;
        ValueToNode[value] = NextID++;
        Gradients[ValueToNode[value]] = 0;
        return ValueToNode[value];
    }
    
    public double RegisterParameter(double value, string name = "")
    {
        Tracked<double> tracked = new()
        {
            Value = value,
            NodeID = NextID++
        };
        Gradients[tracked.NodeID] = 0;
        return tracked;
    }

    private Tracked<double> New(double value, Action<Tracked<double>> backprop)
    {
        Tracked<double> tracked = new() { Value = value, NodeID = NextID++ };
        Tape.Add(new ADNode<double> {ID = tracked.NodeID, Backwards = backprop});
        return tracked;
    }
}

