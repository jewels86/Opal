using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TIn, TOut, TWeightsIn, TWeightsOut>(
    FfLayer<TIn, TOut, TWeightsIn, TWeightsOut> inputLayer,
    List<FfLayer<TOut, TOut, TWeightsOut, TWeightsOut>> hiddenLayers,
    FfLayer<TOut, TOut, TWeightsOut, TWeightsOut> outputLayer,
    Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
    Func<Tensor<TWeightsOut>, Value<TWeightsOut>, Tensor<float>> lossFunctionBatched,
    int hiddenSize,
    Func<Tensor<TOut>, Tensor<TOut>> hiddenActivation,
    Func<Tensor<TWeightsOut>, Tensor<TWeightsOut>> hiddenActivationBatched)
    : INetwork<TIn, TOut>, IBatchingNetwork<TWeightsIn, TWeightsOut>
    where TIn : notnull
    where TOut : notnull
    where TWeightsIn : notnull
    where TWeightsOut : notnull
{
    public FfLayer<TIn, TOut, TWeightsIn, TWeightsOut> InputLayer { get; } = inputLayer;
    public List<FfLayer<TOut, TOut, TWeightsOut, TWeightsOut>> HiddenLayers { get; } = hiddenLayers;
    public FfLayer<TOut, TOut, TWeightsOut, TWeightsOut> OutputLayer { get; } = outputLayer;

    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; } = lossFunction;
    public Func<Tensor<TWeightsOut>, Value<TWeightsOut>, Tensor<float>> LossFunctionBatched { get; } = lossFunctionBatched;

    protected int HiddenSize { get; } = hiddenSize;
    protected Func<Tensor<TOut>, Tensor<TOut>> HiddenActivation { get; } = hiddenActivation;
    protected Func<Tensor<TWeightsOut>, Tensor<TWeightsOut>> HiddenActivationBatched { get; } = hiddenActivationBatched;

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input).Defer();
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current).Defer());
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TWeightsOut> ForwardBatch(Tensor<TWeightsIn> batch)
    {
        var hidden = InputLayer.ForwardBatch(batch).Defer();
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.ForwardBatch(current).Defer());
        return OutputLayer.ForwardBatch(hidden);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;
    public Value<TWeightsOut> ForwardBatch(Value<TWeightsIn> batch) => ForwardBatch(new Tensor<TWeightsIn>(batch, batch.Zeros())).Value;

    public void UpdateParameters(float lr)
    {
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
    }

    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr) => 
        NetworkHelpers.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);

    public void TrainBatches(Value<TWeightsIn>[] inputs, Value<TWeightsOut>[] targets, int epochs, float lr) => 
        NetworkHelpers.Train(ForwardBatch, LossFunctionBatched, () => UpdateParameters(lr), inputs, targets, epochs);
    
    public float EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) =>
        NetworkHelpers.EvaluateLoss(Forward, LossFunction, inputs, targets);
    public float EvaluateLossBatches(Value<TWeightsIn>[] inputs, Value<TWeightsOut>[] targets) =>
        NetworkHelpers.EvaluateLoss(ForwardBatch, LossFunctionBatched, inputs, targets);

    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<TOut, TOut>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<TOut,TOut>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract FfLayer<TOut, TOut, TWeightsOut, TWeightsOut> CreateHiddenLayer();
}