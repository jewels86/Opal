using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Recurrent;

public class RecurrentLayer<TIn, TOut, TWeight> : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull
    where TWeight : notnull 
{
    public required Tensor<TWeight>[] InputWeights { get; set; } 
    public required Tensor<TWeight>[] RecurrentWeights { get; set; }
    public required Tensor<TOut> Biases { get; set; }
    public required Tensor<TOut> State { get; set; }
    public required ActivationFunction<TOut> Activation { get; set; }
    public required IRecurrentCatalog<TIn, TOut, TWeight> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var inputPart = Catalog.Multiply(InputWeights, input);
        var hiddenPart = Catalog.Multiply(RecurrentWeights, State);
        var sum1 = Catalog.Add(inputPart, hiddenPart);
        var sum2 = Catalog.Add(sum1, Biases);
        var output = Activation.Function(sum2);
        State = output;
        return output;
    }

    public TOut Forward(TIn input) => Forward(new Tensor<TIn>(input, null, _ => { }, Catalog.ZeroGradient(input))).Value;

    public void UpdateParameters(double lr)
    {
        foreach (var weight in InputWeights) 
        {
            weight.Value = Catalog.Subtract(weight.Value, Catalog.Scale(weight.Gradient, lr));
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
        }
        foreach (var weight in RecurrentWeights)
        {
            weight.Value = Catalog.Subtract(weight.Value, Catalog.Scale(weight.Gradient, lr));
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
        }
        
        Biases.Value = Catalog.Subtract(Biases.Value, Catalog.Scale(Biases.Gradient, lr));
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void ZeroGradients()
    {
        foreach (var weight in InputWeights)
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
        foreach (var weight in RecurrentWeights)
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(InputWeights.Length);
        foreach (var weight in InputWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(RecurrentWeights.Length);
        foreach (var weight in RecurrentWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        Catalog.WriteBias(writer, Biases.Value);
        Catalog.WriteState(writer, State.Value);
    }

    public void Read(BinaryReader reader)
    {
        int inputWeightCount = reader.ReadInt32();
        InputWeights = new Tensor<TWeight>[inputWeightCount];
        for (int i = 0; i < inputWeightCount; i++)
        {
            var weightValue = Catalog.ReadWeight(reader);
            InputWeights[i] = new Tensor<TWeight>(weightValue, null, _ => { }, Catalog.ZeroGradient(weightValue));
        }
        
        int recurrentWeightCount = reader.ReadInt32();
        RecurrentWeights = new Tensor<TWeight>[recurrentWeightCount];
        for (int i = 0; i < recurrentWeightCount; i++)
        {
            var weightValue = Catalog.ReadWeight(reader);
            RecurrentWeights[i] = new Tensor<TWeight>(weightValue, null, _ => { }, Catalog.ZeroGradient(weightValue));
        }
        
        var biasValue = Catalog.ReadBias(reader);
        Biases = new Tensor<TOut>(biasValue, null, _ => { }, Catalog.ZeroGradient(biasValue));
        
        var stateValue = Catalog.ReadState(reader);
        State = new Tensor<TOut>(stateValue, null, _ => { }, Catalog.ZeroGradient(stateValue));
    }
}

public interface IRecurrentCatalog<TIn, TOut, TWeight>
    where TIn : notnull where TOut : notnull
    where TWeight : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeight>[] weights, Tensor<TIn> input);
    public Tensor<TOut> Multiply(Tensor<TWeight>[] weights, Tensor<TOut> state);
    public Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    public TWeight Subtract(TWeight a, TWeight b);
    public TOut Subtract(TOut a, TOut b);
    
    public TWeight Scale(TWeight a, double scale);
    public TOut Scale(TOut a, double scale);
    
    public TIn ZeroGradient(TIn a);
    public TWeight ZeroGradient(TWeight a);
    public TOut ZeroGradient(TOut a);
    
    public void WriteWeight(BinaryWriter writer, TWeight weight);
    public void WriteBias(BinaryWriter writer, TOut bias);
    public void WriteState(BinaryWriter writer, TOut state);
    
    public TWeight ReadWeight(BinaryReader reader);
    public TOut ReadBias(BinaryReader reader);
    public TOut ReadState(BinaryReader reader);
}