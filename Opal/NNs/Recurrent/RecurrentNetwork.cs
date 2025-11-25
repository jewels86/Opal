using Jewels.Lazulite;
using Opal.Utilities;

namespace Opal.NNs.Recurrent;

public abstract class RecurrentNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut>
    : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
    where THidden : notnull
    where TWeightsIn : notnull
    where TWeightsHidden : notnull
    where TWeightsOut : notnull
{
    protected RecurrentNetwork(int hiddenSize, RecurrentLayer<TIn, THidden, TWeightsIn> inputLayer, List<RecurrentLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        RecurrentLayer<THidden, TOut, TWeightsOut> outputLayer, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction, Func<Tensor<TOut>, Tensor<TOut>> outputActivation, 
        Func<Tensor<THidden>, Tensor<THidden>> hiddenActivation)
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        OutputActivation = outputActivation;
        HiddenActivation = hiddenActivation;
        HiddenSize = hiddenSize;
    }
    
    
    protected int HiddenSize { get; }

    public RecurrentLayer<TIn, THidden, TWeightsIn> InputLayer { get; }
    public List<RecurrentLayer<THidden, THidden, TWeightsHidden>> HiddenLayers { get; }
    public RecurrentLayer<THidden, TOut, TWeightsOut> OutputLayer { get; }
    
    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; }
    public Func<Tensor<TOut>, Tensor<TOut>> OutputActivation { get; }
    public Func<Tensor<THidden>, Tensor<THidden>> HiddenActivation { get; }
    
    public Value<TOut> Forward(Value<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var hidden = InputLayer.Forward(input);
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] sequence) => NetworkHelpers.ForwardSequence(ResetState, Forward, sequence);
    public Value<TOut> ForwardSequence(Value<TIn>[] sequence) => 
        ForwardSequence(sequence.Select(x => new Tensor<TIn>(x, x.Zeros())).ToArray()).Value;

    public void UpdateParameters(float lr)
    {
        InputLayer.UpdateParameters(lr);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(lr);
        OutputLayer.UpdateParameters(lr);
    }

    public void ResetState()
    {
        InputLayer.State = new Tensor<THidden>(InputLayer.State.Value.Zeros(), InputLayer.State.Gradient.Zeros());
        foreach (var layer in HiddenLayers) layer.State = new Tensor<THidden>(layer.State.Value.Zeros(), layer.State.Gradient.Zeros());
        OutputLayer.State = new Tensor<TOut>(OutputLayer.State.Value.Zeros(), OutputLayer.State.Gradient.Zeros());
    }

    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr) =>
        NetworkHelpers.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);

    public float EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) =>
        NetworkHelpers.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr) =>
        NetworkHelpers.TrainSequences(ForwardSequence, LossFunction, ResetState, () => UpdateParameters(lr), sequences, targets, epochs);
    
    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets) =>
        NetworkHelpers.EvaluateLossSequences(ForwardSequence, LossFunction, sequences, targets);
    
    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract RecurrentLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}
