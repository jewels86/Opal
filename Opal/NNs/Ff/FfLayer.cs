using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeights> : ILayer<TIn, TOut> where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public FfLayer(Tensor<TWeights> weights, Tensor<TOut> biases, Func<Tensor<TOut>, Tensor<TOut>> activation, IFfCatalog<TIn, TOut, TWeights> catalog)
    {
        Weights = weights;
        Biases = biases;
        Activation = activation;
        Catalog = catalog;
    }

    public Tensor<TWeights> Weights { get; private set; }
    public Tensor<TOut> Biases { get; private set; }
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; }
    public IFfCatalog<TIn, TOut, TWeights> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var multiplied = Catalog.Multiply(Weights, input);
        var sum = new Tensor<TOut>(Compute.BinaryCall(
                Compute.ElementwiseAddKernels, multiplied.Value, Biases.Value), 
            multiplied.Gradient.Create(Compute.GetLike(multiplied.Gradient), multiplied.Gradient.Shape),
            BackwardFunction,
            [multiplied, Biases]);
        return Activation(sum);

        void BackwardFunction(ITensor t)
        {
            Compute.BinaryCall(Compute.ElementwiseAddKernels, t.Gradient.Data, multiplied.Gradient, multiplied.Gradient);
            Compute.BinaryCall(Compute.ElementwiseAddKernels, t.Gradient.Data, Biases.Gradient, Biases.Gradient);
        }
    }
    public Value<TOut> Forward(Value<TIn> input) => Forward(new(input, input.Create(Compute.GetLike(input), input.Shape))).Value;

    public void UpdateParameters(float lr)
    {
        Compute.Call(Weights.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, Weights.Value, Weights.Value, Weights.Value, lr);
        Compute.Call(Biases.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, Biases.Value, Biases.Value, Biases.Value, lr);
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
}

public interface IFfCatalog<TIn, TOut, TWeights>
    where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TIn> b);
    
    public void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    public Value<TWeights> ReadWeights(BinaryReader reader);
    public void WriteBias(BinaryWriter writer, Value<TOut> bias);
    public Value<TOut> ReadBias(BinaryReader reader);
}