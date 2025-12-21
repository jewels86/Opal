using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public abstract class LstmNetwork<TIn, TOut, TWeightsIn, TWeightsOut, TBiasesIn, TBiasesOut>(
    LstmLayer<TIn, TOut, TWeightsIn, TBiasesIn> inputLayer,
    List<LstmLayer<TOut, TOut, TWeightsOut, TBiasesOut>> hiddenLayers,
    LstmLayer<TOut, TOut, TWeightsOut, TBiasesOut> outputLayer,
    Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
    int hiddenSize)
    : ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
    where TWeightsIn : notnull
    where TWeightsOut : notnull
    where TBiasesIn : notnull
    where TBiasesOut : notnull
{
    public LstmLayer<TIn, TOut, TWeightsIn, TBiasesIn> InputLayer { get; set; } = inputLayer;
    public List<LstmLayer<TOut, TOut, TWeightsOut, TBiasesOut>> HiddenLayers { get; set; } = hiddenLayers;
    public LstmLayer<TOut, TOut, TWeightsOut, TBiasesOut> OutputLayer { get; set; } = outputLayer;
    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; } = lossFunction;
    
    public float? DefaultGradClipNorm { get; set; } = null;

    protected int HiddenSize { get; } = hiddenSize;

    public Value<TOut> Forward(Value<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] sequence) => Operations.ForwardSequence(ResetState, Forward, sequence);

    public Value<TOut> ForwardSequence(Value<TIn>[] sequence) => 
        ForwardSequence(sequence.Select(i => new Tensor<TIn>(i, i.Zeros())).ToArray()).Value;

    public void UpdateParameters(float lr, float? gradClipNorm = null, List<ITensor>? clipTensors = null, bool reset = true)
    {
        //Console.WriteLine($"Before update - weight sample: {InputLayer.EncoderForgetWeights.Value.ToProxy().FlatData[0]}");

        if (clipTensors is not null && gradClipNorm.HasValue)
            Operations.ClipGradientsByNorm(gradClipNorm.Value, clipTensors.ToArray());
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
        //Console.WriteLine($"After update - weight sample: {InputLayer.EncoderForgetWeights.Value.ToProxy().FlatData[0]}");

        if (reset) ResetState();
    }

    public void ResetState()
    {
        InputLayer.ResetState();
        foreach (var layer in HiddenLayers) layer.ResetState();
        OutputLayer.ResetState();
    }

    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr) =>
        Operations.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);

    public double EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) => Operations.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr)
    {
        List<ITensor> tensors = [];
        tensors.AddRange(InputLayer.Parameters);
        foreach (var hidden in HiddenLayers) tensors.AddRange(hidden.Parameters);
        tensors.AddRange(OutputLayer.Parameters);
        
        Operations.TrainSequencesFinal(ForwardSequence, LossFunction, ResetState, () => UpdateParameters(lr, DefaultGradClipNorm, tensors), sequences, targets, epochs);
    }

    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets) =>
        Operations.EvaluateLossSequences(ForwardSequence, LossFunction, sequences, targets);
    
    public void Save(string path) => Operations.Save(InputLayer, HiddenLayers.Cast<ILayer<TOut,TOut>>().ToList(), OutputLayer, path);
    public void Load(string path) => Operations.Load(InputLayer, HiddenLayers.Cast<ILayer<TOut,TOut>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract LstmLayer<TOut, TOut, TWeightsOut, TBiasesOut> CreateHiddenLayer();
}
