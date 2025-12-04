using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

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
    
    public float? DefaultGradClipNorm { get; set; } = null;
    public float DefaultTrainingEpsilon { get; set; } = 1e-4f;
    public float DefaultInitialGradient { get; set; } = 1;
    public int DefaultCheckInterval { get; set; } = 100;

    protected int HiddenSize { get; } = hiddenSize;
    protected Func<Tensor<TOut>, Tensor<TOut>> HiddenActivation { get; } = hiddenActivation;

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers) hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;

    public void UpdateParameters(float lr, float? gradClipNorm = null, List<ITensor>? clipTensors = null)
    {
        if (clipTensors is not null && gradClipNorm.HasValue)
            Operations.ClipGradientsByNorm(gradClipNorm.Value, clipTensors.ToArray());
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
    }

    public List<float> Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr)
    {
        List<ITensor> tensors = [InputLayer.Weights, InputLayer.Biases, OutputLayer.Weights, OutputLayer.Biases];
        tensors.AddRange(HiddenLayers.Select(layer => layer.Weights));
        tensors.AddRange(HiddenLayers.Select(layer => layer.Biases));
        return Operations.Train(
            Forward,
            LossFunction,
            () => UpdateParameters(lr, DefaultGradClipNorm, tensors),
            inputs, targets, epochs,
            DefaultTrainingEpsilon, DefaultCheckInterval, DefaultInitialGradient);
    }

    public float EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) =>
        Operations.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void Save(string path) => Operations.Save(InputLayer, HiddenLayers.Cast<ILayer<TOut, TOut>>().ToList(), OutputLayer, path);
    public void Load(string path) => Operations.Load(InputLayer, HiddenLayers.Cast<ILayer<TOut,TOut>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract FfLayer<TOut, TOut, TWeightsOut, TBiasesOut> CreateHiddenLayer();
    
    public void Dispose()
    {
        InputLayer.Dispose();
        foreach (var layer in HiddenLayers) layer.Dispose();
        OutputLayer.Dispose();
    }
}