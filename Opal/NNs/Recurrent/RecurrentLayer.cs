using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Recurrent;

public class RecurrentLayer<TIn, TOut, TWeights> : ILayer<TIn, TOut>
    where TIn : notnull, IDisposable where TOut : notnull, IDisposable
    where TWeights : notnull, IDisposable
{
    public RecurrentLayer(
        Tensor<TWeights> inputWeights, Tensor<TWeights> recurrentWeights, 
        Tensor<TOut> biases, Tensor<TOut> state, 
        Func<Tensor<TOut>, Tensor<TOut>> activation, IRecurrentCatalog<TIn, TOut, TWeights> catalog)
    {
        InputWeights = inputWeights;
        RecurrentWeights = recurrentWeights;
        Biases = biases;
        State = state;
        Activation = activation;
        Catalog = catalog;
    }

    public Tensor<TWeights> InputWeights { get; set; } 
    public Tensor<TWeights> RecurrentWeights { get; set; }
    public Tensor<TOut> Biases { get; set; }
    public Tensor<TOut> State { get; set; }
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; }
    public IRecurrentCatalog<TIn, TOut, TWeights> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        using var inputPart = Catalog.Multiply(InputWeights, input);
        using var hiddenPart = Catalog.Multiply(RecurrentWeights, State);
        using var sum1 = Catalog.Add(inputPart, hiddenPart);
        using var sum2 = Catalog.Add(sum1, Biases);
        var output = Activation(sum2);
        State = output;
        return output;
    }

    public TOut Forward(TIn input) => Forward(new Tensor<TIn>(input, null, _ => { }, Catalog.ZeroGradient(input))).Value;

    public void UpdateParameters(double lr)
    {
        InputWeights.Value = Catalog.Subtract(InputWeights.Value, Catalog.Scale(InputWeights.Gradient, lr).Defer());
        InputWeights.Gradient = Catalog.ZeroGradient(InputWeights.Value);
        
        RecurrentWeights.Value = Catalog.Subtract(RecurrentWeights.Value, Catalog.Scale(RecurrentWeights.Gradient, lr).Defer());
        RecurrentWeights.Gradient = Catalog.ZeroGradient(RecurrentWeights.Value);
        
        Biases.Value = Catalog.Subtract(Biases.Value, Catalog.Scale(Biases.Gradient, lr).Defer());
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void ZeroGradients()
    {
        InputWeights.Gradient = Catalog.ZeroGradient(InputWeights.Value);
        RecurrentWeights.Gradient = Catalog.ZeroGradient(RecurrentWeights.Value);
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void Write(BinaryWriter writer)
    {
        Catalog.WriteWeights(writer, InputWeights.Value);
        Catalog.WriteWeights(writer, RecurrentWeights.Value);
        Catalog.WriteBias(writer, Biases.Value);
        Catalog.WriteState(writer, State.Value);
    }

    public void Read(BinaryReader reader)
    {
        InputWeights = new Tensor<TWeights>(Catalog.ReadWeights(reader), null, _ => { }, Catalog.ZeroGradient(InputWeights.Value));
        RecurrentWeights = new Tensor<TWeights>(Catalog.ReadWeights(reader), null, _ => { }, Catalog.ZeroGradient(RecurrentWeights.Value));
        
        var biasValue = Catalog.ReadBias(reader);
        Biases = new Tensor<TOut>(biasValue, null, _ => { }, Catalog.ZeroGradient(biasValue));
        
        var stateValue = Catalog.ReadState(reader);
        State = new Tensor<TOut>(stateValue, null, _ => { }, Catalog.ZeroGradient(stateValue));
    }
}

public interface IRecurrentCatalog<TIn, TOut, TWeights>
    where TIn : notnull where TOut : notnull
    where TWeights : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> weights, Tensor<TIn> input);
    public Tensor<TOut> Multiply(Tensor<TWeights> weights, Tensor<TOut> state);
    public Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    public TWeights Subtract(TWeights a, TWeights b);
    public TOut Subtract(TOut a, TOut b);
    
    public TWeights Scale(TWeights a, double scale);
    public TOut Scale(TOut a, double scale);
    
    public TIn ZeroGradient(TIn a);
    public TWeights ZeroGradient(TWeights a);
    public TOut ZeroGradient(TOut a);
    
    public void WriteWeights(BinaryWriter writer, TWeights weight);
    public void WriteBias(BinaryWriter writer, TOut bias);
    public void WriteState(BinaryWriter writer, TOut state);
    
    public TWeights ReadWeights(BinaryReader reader);
    public TOut ReadBias(BinaryReader reader);
    public TOut ReadState(BinaryReader reader);
}