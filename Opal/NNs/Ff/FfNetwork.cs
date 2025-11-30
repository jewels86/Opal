using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TIn, TOut, TWeightsIn, TWeightsOut, TBiasesIn, TBiasesOut>(
    FfLayer<TIn, TOut, TWeightsIn, TBiasesIn> inputLayer,
    List<FfLayer<TOut, TOut, TWeightsOut, TBiasesOut>> hiddenLayers,
    FfLayer<TOut, TOut, TWeightsOut, TBiasesOut> outputLayer,
    Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
    int hiddenSize,
    Func<Tensor<TOut>, Tensor<TOut>> hiddenActivation)
    : INetwork<TIn, TOut>, IDisposable
    where TIn : notnull
    where TOut : notnull
    where TWeightsIn : notnull
    where TWeightsOut : notnull
    where TBiasesIn : notnull
    where TBiasesOut : notnull
{
    public FfLayer<TIn, TOut, TWeightsIn, TBiasesIn> InputLayer { get; } = inputLayer;
    public List<FfLayer<TOut, TOut, TWeightsOut, TBiasesOut>> HiddenLayers { get; } = hiddenLayers;
    public FfLayer<TOut, TOut, TWeightsOut, TBiasesOut> OutputLayer { get; } = outputLayer;

    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; } = lossFunction;

    protected int HiddenSize { get; } = hiddenSize;
    protected Func<Tensor<TOut>, Tensor<TOut>> HiddenActivation { get; } = hiddenActivation;

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers) hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;

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
    protected abstract FfLayer<TOut, TOut, TWeightsOut, TBiasesOut> CreateHiddenLayer();
    
    public void Dispose()
    {
        InputLayer.Dispose();
        foreach (var layer in HiddenLayers) layer.Dispose();
        OutputLayer.Dispose();
    }
}