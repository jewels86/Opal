using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeights, TBatchOut> : IBatchedLayer<TIn, TOut, TWeights, TBatchOut>, IDisposable
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBatchOut : notnull
{
    public FfLayer(
        Tensor<TWeights> weights, 
        Tensor<TOut> biases, 
        Func<Tensor<TOut>, Tensor<TOut>> activation, 
        Func<Tensor<TBatchOut>, Tensor<TBatchOut>> batchedActivation,
        IFfCatalog<TIn, TOut, TWeights, TBatchOut> catalog)
    {
        Weights = weights;
        Biases = biases;
        Activation = activation;
        BatchedActivation = batchedActivation;
        Catalog = catalog;
    }

    public Tensor<TWeights> Weights { get; private set; }
    public Tensor<TOut> Biases { get; private set; }
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; }
    public Func<Tensor<TBatchOut>, Tensor<TBatchOut>> BatchedActivation { get; set; }
    public IFfCatalog<TIn, TOut, TWeights, TBatchOut> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var multiplied = Catalog.Multiply(Weights, input, disposeA: false);
        var sum = Operations.Add(Biases, multiplied, disposeA: false);
        return Activation(sum);
    }
    public Tensor<TBatchOut> ForwardBatch(Tensor<TWeights> batch)
    {
        var multiplied = Catalog.Multiply(batch, Weights, disposeB: false);
        var sum = Catalog.Add(multiplied, Biases, disposeB: false);
        return BatchedActivation(sum);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;
    public Value<TBatchOut> ForwardBatch(Value<TWeights> batch) => ForwardBatch(new Tensor<TWeights>(batch, batch.Zeros())).Value;
    
    public void UpdateParameters(float lr)
    {
        Compute.Call(Weights.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, Weights.Gradient, Weights.Value, Weights.Value, lr);
        Compute.Call(Biases.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, Biases.Gradient, Biases.Value, Biases.Value, lr);
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
    public Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TIn> b, bool disposeA = true, bool disposeB = true);
    public Tensor<TBatchOut> Multiply(Tensor<TWeights> a, Tensor<TWeights> b, bool disposeA = true, bool disposeB = true);
    public Tensor<TBatchOut> Add(Tensor<TBatchOut> a, Tensor<TOut> b, bool disposeA = true, bool disposeB = true);
    
    public void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    public Value<TWeights> ReadWeights(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, Value<TOut> bias);
    public Value<TOut> ReadBias(BinaryReader reader);
}