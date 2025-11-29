using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeights, TBatchOut>(Tensor<TWeights> weights,
    Tensor<TOut> biases,
    Func<Tensor<TOut>, Tensor<TOut>> activation,
    Func<Tensor<TBatchOut>, Tensor<TBatchOut>> batchActivation,
    IFfCatalog<TIn, TOut, TWeights, TBatchOut> catalog)
    : ILayer<TIn, TOut>, IDisposable
    where TIn : notnull
    where TOut : notnull
    where TWeights : notnull
    where TBatchOut : notnull
{

    public Tensor<TWeights> Weights { get; private set; } = weights;
    public Tensor<TOut> Biases { get; private set; } = biases;
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; } = activation;
    public Func<Tensor<TBatchOut>, Tensor<TBatchOut>> BatchActivation { get; set; } = batchActivation;
    public IFfCatalog<TIn, TOut, TWeights, TBatchOut> Catalog { get; set; } = catalog;

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        using var multiplied = Catalog.Multiply(Weights, input);
        using var sum = Operations.Add(Biases, multiplied);
        return Activation(sum);
    }

    public Tensor<TBatchOut> ForwardBatch(Tensor<TWeights> batch)
    {
        var multiplied = Catalog.Multiply(batch, Weights);
        var sum = Catalog.Add(Biases, multiplied);
        return BatchActivation(sum);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;
    
    public void UpdateParameters(float lr)
    {
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
    
    ~FfLayer() => Dispose();
}

public interface IFfCatalog<TIn, TOut, TWeights, TBatchOut>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBatchOut : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TIn> b);
    public Tensor<TBatchOut> Multiply(Tensor<TWeights> a, Tensor<TWeights> b);
    public Tensor<TBatchOut> Add(Tensor<TOut> a, Tensor<TBatchOut> b);
    
    public void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    public Value<TWeights> ReadWeights(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, Value<TOut> bias);
    public Value<TOut> ReadBias(BinaryReader reader);
}