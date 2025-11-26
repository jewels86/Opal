using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut>
    : INetwork<TIn, TOut>
    where TIn : notnull where TOut : notnull where THidden : notnull
    where TWeightsIn : notnull where TWeightsHidden : notnull where TWeightsOut : notnull
{
    public FfLayer<TIn, THidden, TWeightsIn> InputLayer { get; }
    public List<FfLayer<THidden, THidden, TWeightsHidden>> HiddenLayers { get; }
    public FfLayer<THidden, TOut, TWeightsOut> OutputLayer { get; }
    
    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected Func<Tensor<THidden>, Tensor<THidden>> HiddenActivation { get; }
    
    protected FfNetwork(
        FfLayer<TIn, THidden, TWeightsIn> inputLayer,
        List<FfLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        FfLayer<THidden, TOut, TWeightsOut> outputLayer,
        Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
        int hiddenSize,
        Func<Tensor<THidden>, Tensor<THidden>> hiddenActivation)
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
        HiddenActivation = hiddenActivation;
    }

    public Value<TOut> Forward(Value<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        Tensor<THidden> hidden = InputLayer.Forward(input);
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
        return OutputLayer.Forward(hidden);
    }

    protected void UpdateParameters(float lr)
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

    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract FfLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}