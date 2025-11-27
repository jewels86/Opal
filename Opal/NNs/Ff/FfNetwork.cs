using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TIn, TOut, TWeightsIn, TWeightsOut>(
    FfLayer<TIn, TOut, TWeightsIn> inputLayer,
    List<FfLayer<TOut, TOut, TWeightsOut>> hiddenLayers,
    FfLayer<TOut, TOut, TWeightsOut> outputLayer,
    Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
    int hiddenSize,
    Func<Tensor<TOut>, Tensor<TOut>> hiddenActivation)
    : INetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
    where TWeightsIn : notnull
    where TWeightsOut : notnull
{
    public FfLayer<TIn, TOut, TWeightsIn> InputLayer { get; } = inputLayer;
    public List<FfLayer<TOut, TOut, TWeightsOut>> HiddenLayers { get; } = hiddenLayers;
    public FfLayer<TOut, TOut, TWeightsOut> OutputLayer { get; } = outputLayer;

    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; } = lossFunction;

    protected int HiddenSize { get; } = hiddenSize;
    protected Func<Tensor<TOut>, Tensor<TOut>> HiddenActivation { get; } = hiddenActivation;

    public Value<TOut> Forward(Value<TIn> input)
    {
        var hidden = InputLayer.Forward(input).Defer();
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current).Defer());
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input).Defer();
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current).Defer());
        return OutputLayer.Forward(hidden);
    }

    public void UpdateParameters(float lr)
    {
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
    }

    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr) => 
        NetworkHelpers.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);


    public float EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) =>
        NetworkHelpers.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<TOut, TOut>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<TOut,TOut>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract FfLayer<TOut, TOut, TWeightsOut> CreateHiddenLayer();
}