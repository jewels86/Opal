using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeights, TBiases>(
    Tensor<TWeights> weights,
    Tensor<TBiases> biases,
    Func<Tensor<TOut>, Tensor<TOut>> activation,
    IFfCatalog<TIn, TOut, TWeights, TBiases> catalog)
    : ILayer<TIn, TOut>, IDisposable
    where TIn : notnull
    where TOut : notnull
    where TWeights : notnull
    where TBiases : notnull
{

    public Tensor<TWeights> Weights { get; private set; } = weights;
    public Tensor<TBiases> Biases { get; private set; } = biases;
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; } = activation;
    public IFfCatalog<TIn, TOut, TWeights, TBiases> Catalog { get; set; } = catalog;

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var multiplied = Catalog.Multiply(Weights, input);
        var sum = Catalog.Add(Biases, multiplied);
        return Activation(sum);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;
    
    public void UpdateParameters(float lr, float? gradClipNorm = null)
    {
        if (gradClipNorm.HasValue) Operations.ClipGradientsByNorm(gradClipNorm.Value, Weights, Biases);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, Weights.Gradient, Weights.Value, Weights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, Biases.Gradient, Biases.Value, Biases.Value, lr);
        ZeroGradients();
    }
    
    public void ZeroGradients()
    {
        Weights.Gradient.UpdateWith(Weights.Gradient.Zeros());
        Biases.Gradient.UpdateWith(Biases.Gradient.Zeros());
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
        Weights = new(weightsValue, weightsValue.Zeros());
        Biases = new(biasValue, biasValue.Zeros());
    }
    
    public void Dispose()
    {
        Weights.Dispose();
        Biases.Dispose();
    }
}

public interface IFfCatalog<TIn, TOut, TWeights, TBiases>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TIn> b);
    public Tensor<TOut> Add(Tensor<TBiases> a, Tensor<TOut> b);
    
    public void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    public Value<TWeights> ReadWeights(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, Value<TBiases> bias);
    public Value<TBiases> ReadBias(BinaryReader reader);
}