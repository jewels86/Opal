using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeights>  : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public Tensor<TWeights> Weights { get; set; }
    public Tensor<TOut> Biases { get; set; }
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; }
    public IFfCatalog<TIn, TOut, TWeights> Catalog { get; set; }

    public FfLayer(Tensor<TWeights> weights, Tensor<TOut> biases, Func<Tensor<TOut>, Tensor<TOut>> activation, IFfCatalog<TIn, TOut, TWeights> catalog)
    {
        Weights = weights;
        Biases = biases;
        Activation = activation;
        Catalog = catalog;
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        Tensor<TOut> weightedSum = Catalog.Multiply(Weights, input);
        Tensor<TOut> preActivation = Catalog.Add(weightedSum, Biases);
        Tensor<TOut> output = Activation(preActivation);
        return output;
    }
    public TOut Forward(TIn input) => Forward(new Tensor<TIn>(input, null, _ => { }, Catalog.ZeroGradient(input))).Value;

    public void UpdateParameters(double lr)
    {
        Weights.Value = Catalog.Subtract(Weights.Value, Catalog.Scale(Weights.Gradient, lr));
        Weights.Gradient = Catalog.ZeroGradient(Weights.Value);
        
        Biases.Value = Catalog.Subtract(Biases.Value, Catalog.Scale(Biases.Gradient, lr));
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }
    
    public void ZeroGradients()
    {
        Weights.Gradient = Catalog.ZeroGradient(Weights.Value);
        Biases.Gradient = Catalog.ZeroGradient(Biases.Value);
    }

    public void Write(BinaryWriter writer)
    {
        Catalog.WriteWeights(writer, Weights.Value);
        Catalog.WriteBias(writer, Biases.Value);
    }

    public void Read(BinaryReader reader)
    {
        var weightsValue = Catalog.ReadWeights(reader);
        var biasValue = Catalog.ReadBias(reader);
        Weights = new Tensor<TWeights>(weightsValue, null, _ => { }, Catalog.ZeroGradient(weightsValue));
        Biases = new Tensor<TOut>(biasValue, null, _ => { }, Catalog.ZeroGradient(biasValue));
    }
}

public interface IFfCatalog<TIn, TOut, TWeights>
    where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TIn> b);
    public Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    public TWeights Subtract(TWeights a, TWeights b);
    public TOut Subtract(TOut a, TOut b);
    public TWeights Scale(TWeights a, double scale);
    public TOut Scale(TOut a, double scale);
    
    public TOut ZeroGradient(TOut a);
    public TIn ZeroGradient(TIn a);
    public TWeights ZeroGradient(TWeights a);
    
    public void WriteWeights(BinaryWriter writer, TWeights weight);
    public TWeights ReadWeights(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, TOut bias);
    public TOut ReadBias(BinaryReader reader);
}