using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class FfLayer<TIn, TOut, TWeight>
{
    public Tensor<TWeight>[] Weights { get; set; }
    public Tensor<TOut> Biases { get; set; }
    public ActivationFunction<Tensor<TOut>> Activation { get; set; }
    public IFfCatalog<TIn, TOut, TWeight> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        Tensor<TOut> weightedSum = Catalog.Multiply(input, Weights);
        Tensor<TOut> preActivation = Catalog.Sum(weightedSum, Biases);
        Tensor<TOut> output = Activation.Function(preActivation);
        return output;
    }
}

public interface IFfCatalog<TIn, TOut, TWeight>
{
    public Tensor<TOut> Multiply(Tensor<TIn> a, Tensor<TWeight>[] b);
    public Tensor<TOut> Sum(params Tensor<TOut>[] tensors);
}