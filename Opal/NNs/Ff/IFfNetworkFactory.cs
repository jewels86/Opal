using Opal.Mathematics;

namespace Opal.NNs.Ff;

public interface IFfNetworkFactory<TWeights, TBiases, TInput, THidden, TOutput, TFfNetwork>
    where TWeights : notnull
    where TBiases : notnull
    where TInput : notnull
    where THidden : notnull
    where TOutput : notnull
    where TFfNetwork : FfNetwork<TWeights, TBiases, TInput, THidden, TOutput>
{
    public TFfNetwork Create(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<THidden> hiddenActivation, ActivationFunction<TOutput> outputActivation,
        LossFunction<TOutput> lossFunction, IOptimizer<TWeights, TBiases> optimizer, string name = "FfNetwork");
}