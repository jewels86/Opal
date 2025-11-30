using Jewels.Lazulite;

namespace Opal.NNs;

public abstract class RecurrentNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut>(
    int hiddenSize, 
    RecurrentLayer<TIn, THidden, TWeightsIn> inputLayer,
    List<RecurrentLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
    RecurrentLayer<THidden, TOut, TWeightsOut> outputLayer, 
    Func<Tensor<TOut>, Value<TOut>, Tensor<float>> lossFunction, 
    Func<Tensor<TOut>, Tensor<TOut>> outputActivation,
    Func<Tensor<THidden>, Tensor<THidden>> hiddenActivation)
    : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
    where THidden : notnull
    where TWeightsIn : notnull
    where TWeightsHidden : notnull
    where TWeightsOut : notnull
{


    protected int HiddenSize { get; } = hiddenSize;

    public RecurrentLayer<TIn, THidden, TWeightsIn> InputLayer { get; } = inputLayer;
    public List<RecurrentLayer<THidden, THidden, TWeightsHidden>> HiddenLayers { get; } = hiddenLayers;
    public RecurrentLayer<THidden, TOut, TWeightsOut> OutputLayer { get; } = outputLayer;

    public Func<Tensor<TOut>, Value<TOut>, Tensor<float>> LossFunction { get; } = lossFunction;
    public Func<Tensor<TOut>, Tensor<TOut>> OutputActivation { get; } = outputActivation;
    public Func<Tensor<THidden>, Tensor<THidden>> HiddenActivation { get; } = hiddenActivation;

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

    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] sequence) => Operations.ForwardSequence(ResetState, Forward, sequence);
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
        Operations.Train(Forward, LossFunction, () => UpdateParameters(lr), inputs, targets, epochs);

    public float EvaluateLoss(Value<TIn>[] inputs, Value<TOut>[] targets) =>
        Operations.EvaluateLoss(Forward, LossFunction, inputs, targets);

    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr) =>
        Operations.TrainSequences(ForwardSequence, LossFunction, ResetState, () => UpdateParameters(lr), sequences, targets, epochs);
    
    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets) =>
        Operations.EvaluateLossSequences(ForwardSequence, LossFunction, sequences, targets);
    
    public void Save(string path) => Operations.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => Operations.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract RecurrentLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}
