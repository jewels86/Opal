using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeight>  : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeight : notnull
{
    public Tensor<TWeight>[] Weights { get; set; }
    public Tensor<TOut> Biases { get; set; }
    public ActivationFunction<TOut> Activation { get; set; }
    public IFfCatalog<TIn, TOut, TWeight> Catalog { get; set; }

    public FfLayer(Tensor<TWeight>[] weights, Tensor<TOut> biases, ActivationFunction<TOut> activation, IFfCatalog<TIn, TOut, TWeight> catalog)
    {
        Weights = weights;
        Biases = biases;
        Activation = activation;
        Catalog = catalog;
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        Tensor<TOut> weightedSum = Catalog.Multiply(input, Weights);
        Tensor<TOut> preActivation = Catalog.Add(weightedSum, Biases);
        Tensor<TOut> output = Activation.Function(preActivation);
        return output;
    }
    public TOut Forward(TIn input) => Forward(new Tensor<TIn>(input, null, _ => { }, Catalog.ZeroGradient(input))).Value;

    public void UpdateParameters(double lr)
    {
        foreach (var weight in Weights)
        {
            weight.Value = Catalog.Subtract(weight.Value, Catalog.Scale(weight.Gradient, lr));
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
        }
        
        Biases.Value = Catalog.Subtract(Biases.Value, Catalog.Scale(Biases.Gradient, lr));
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }
    
    public void ZeroGradients()
    {
        foreach (var weight in Weights)
            weight.Gradient = Catalog.ZeroGradient(weight.Value);
    
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(Weights.Length);
        foreach (var weight in Weights)
            Catalog.WriteWeight(writer, weight.Value);
        Catalog.WriteBias(writer, Biases.Value);
    }

    public void Read(BinaryReader reader)
    {
        int weightCount = reader.ReadInt32();
        Weights = new Tensor<TWeight>[weightCount];
        for (int i = 0; i < weightCount; i++)
        {
            var weightValue = Catalog.ReadWeight(reader);
            Weights[i] = new Tensor<TWeight>(weightValue, null, _ => { }, Catalog.ZeroGradient(weightValue));
        }
        var biasValue = Catalog.ReadBias(reader);
        Biases = new Tensor<TOut>(biasValue, null, _ => { }, Catalog.ZeroGradient(biasValue));
    }
}

public interface IFfCatalog<TIn, TOut, TWeight>
    where TIn : notnull where TOut : notnull where TWeight : notnull
{
    public Tensor<TOut> Multiply(Tensor<TIn> a, Tensor<TWeight>[] b);
    public Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    
    public TWeight Subtract(TWeight a, TWeight b);
    public TWeight Scale(TWeight a, double scale);
    public TWeight ZeroGradient(TWeight a);
    
    public TOut Subtract(TOut a, TOut b);
    public TOut Scale(TOut a, double scale);
    public TOut ZeroGradient(TOut a);
    
    public TIn ZeroGradient(TIn a);
    
    public void WriteWeight(BinaryWriter writer, TWeight weight);
    public TWeight ReadWeight(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, TOut bias);
    public TOut ReadBias(BinaryReader reader);
}