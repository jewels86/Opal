using Jewels.Lazulite;

namespace Opal.NNs.Lstm;

public abstract class LstmNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut> : ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where THidden : notnull
    where TOut : notnull
    where TWeightsIn : notnull
    where TWeightsHidden : notnull
    where TWeightsOut : notnull
{
    public LstmLayer<TIn, THidden, TWeightsIn> InputLayer { get; set; }
    public List<LstmLayer<THidden, THidden, TWeightsHidden>> HiddenLayers { get; set; }
    public LstmLayer<THidden, TOut, TWeightsOut> OutputLayer { get; set; }
    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; }
    
    protected int HiddenSize { get; }
    
    protected LstmNetwork(
        LstmLayer<TIn, THidden, TWeightsIn> inputLayer,
        List<LstmLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        LstmLayer<THidden, TOut, TWeightsOut> outputLayer,
        Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction,
        int hiddenSize)
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
    }
    
    public Value<TOut> Forward(Value<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        Tensor<THidden> hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] sequence) => NetworkHelpers.ForwardSequence(() => { }, Forward, sequence);

    public Value<TOut> ForwardSequence(Value<TIn>[] sequence) => 
        ForwardSequence(sequence.Select(i => new Tensor<TIn>(i, i.Zeros())).ToArray()).Value;

    public void UpdateParameters(float lr)
    {
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
    }

    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr) =>
        NetworkHelpers.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);

    public double EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) => NetworkHelpers.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr) =>
        NetworkHelpers.TrainSequences(ForwardSequence, LossFunction, () => { }, () => UpdateParameters(lr), sequences, targets, epochs);
    
    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets) =>
        NetworkHelpers.EvaluateLossSequences(ForwardSequence, LossFunction, sequences, targets);
    
    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract LstmLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}
