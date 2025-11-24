using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Recurrent;

public abstract class RecurrentNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut>
    : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull, IDisposable
    where TOut : notnull, IDisposable
    where THidden : notnull, IDisposable
    where TWeightsIn : notnull, IDisposable
    where TWeightsHidden : notnull, IDisposable
    where TWeightsOut : notnull, IDisposable
{
    protected RecurrentNetwork(int hiddenSize, RecurrentLayer<TIn, THidden, TWeightsIn> inputLayer, List<RecurrentLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        RecurrentLayer<THidden, TOut, TWeightsOut> outputLayer, Func<Tensor<TOut>, TOut, ScalarTensor> lossFunction, Func<Tensor<TOut>, Tensor<TOut>> outputActivation, 
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
    
    public Func<Tensor<TOut>, TOut, ScalarTensor> LossFunction { get; }
    public Func<Tensor<TOut>, Tensor<TOut>> OutputActivation { get; }
    public Func<Tensor<THidden>, Tensor<THidden>> HiddenActivation { get; }
    
    public TOut Forward(TIn input)
    {
        THidden hidden = InputLayer.Forward(input);
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

    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] sequence) =>
        NetworkHelpers.ForwardSequence(ResetState, Forward, sequence);

    public TOut ForwardSequence(TIn[] sequence) => 
        ForwardSequence(
            sequence.Select(i => new Tensor<TIn>(i, null, _ => { }, 
                InputLayer.Catalog.ZeroGradient(i))).ToArray()).Value;

    public void UpdateParameters(double learningRate)
    {
        InputLayer.UpdateParameters(learningRate);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(learningRate);
        OutputLayer.UpdateParameters(learningRate);
    }

    public void ResetState()
    {
        var inputZero = InputLayer.Catalog.ZeroGradient(InputLayer.State.Value);
        InputLayer.State = new Tensor<THidden>(
            inputZero,
            null,
            _ => { },
            inputZero);
        
        foreach (var layer in HiddenLayers)
        {
            layer.State = new Tensor<THidden>(
                inputZero,
                null,
                _ => { },
                inputZero);
        }
        
        
        OutputLayer.State = new Tensor<TOut>(
            OutputLayer.Catalog.ZeroGradient(OutputLayer.State.Value),
            null,
            _ => { },
            OutputLayer.Catalog.ZeroGradient(OutputLayer.State.Value));
    }

    public void Train(TIn[] inputs, TOut[] targets, int epochs, double learningRate) =>
        NetworkHelpers.Train(
            i => InputLayer.Catalog.ZeroGradient(i), 
            Forward, LossFunction, 
            () => UpdateParameters(learningRate), 
            inputs, targets, epochs);

    public double EvaluateLoss(TIn[] inputs, TOut[] targets) =>
        NetworkHelpers.EvaluateLoss(
            i => InputLayer.Catalog.ZeroGradient(i), 
            Forward, LossFunction, 
            inputs, targets);

    public void TrainSequences(TIn[][] sequences, TOut[] targets, int epochs, double learningRate) =>
        NetworkHelpers.TrainSequences(
            i => InputLayer.Catalog.ZeroGradient(i), Forward, LossFunction, 
            () => UpdateParameters(learningRate), ResetState,
            sequences, targets, epochs);
    
    public double EvaluateLossSequences(TIn[][] sequences, TOut[] targets) =>
        NetworkHelpers.EvaluateLossSequences(
            i => InputLayer.Catalog.ZeroGradient(i), Forward, LossFunction, ResetState,
            sequences, targets);
    
    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract RecurrentLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}
