using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public abstract class LstmNetwork<TIn, THidden, TOut, TWeightsIn, TWeightsHidden, TWeightsOut> : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
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
    public Func<Tensor<TOut>, TOut, ScalarTensor> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected Func<Tensor<TOut>, Tensor<TOut>> SigmoidOutActivation { get; }
    protected Func<Tensor<TOut>, Tensor<TOut>> TanhOutActivation { get; }
    protected Func<Tensor<THidden>, Tensor<THidden>> TanhHiddenActivation { get; }
    protected Func<Tensor<THidden>, Tensor<THidden>> SigmoidHiddenActivation { get; }
    
    
    protected LstmNetwork(
        LstmLayer<TIn, THidden, TWeightsIn> inputLayer,
        List<LstmLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        LstmLayer<THidden, TOut, TWeightsOut> outputLayer,
        Func<Tensor<TOut>, TOut, ScalarTensor> lossFunction,
        int hiddenSize,
        Func<Tensor<TOut>, Tensor<TOut>> sigmoidOutActivation,
        Func<Tensor<TOut>, Tensor<TOut>> tanhOutActivation,
        Func<Tensor<THidden>, Tensor<THidden>> tanhHiddenActivation,
        Func<Tensor<THidden>, Tensor<THidden>> sigmoidHiddenActivation)
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
        SigmoidOutActivation = sigmoidOutActivation;
        TanhOutActivation = tanhOutActivation;
        TanhHiddenActivation = tanhHiddenActivation;
        SigmoidHiddenActivation = sigmoidHiddenActivation;
    }
    
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
        NetworkHelpers.ForwardSequence(() => { }, Forward, sequence);

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
            () => UpdateParameters(learningRate), () => { },
            sequences, targets, epochs);
    
    public double EvaluateLossSequences(TIn[][] sequences, TOut[] targets) =>
        NetworkHelpers.EvaluateLossSequences(
            i => InputLayer.Catalog.ZeroGradient(i), Forward, LossFunction, () => { },
            sequences, targets);
    
    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract LstmLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}
