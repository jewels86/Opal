using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TInput, THidden, TOutput, TWeightsIn, TWeightsHidden, TWeightsOut>
    : INetwork<TInput, TOutput>
    where TInput : notnull, IDisposable where TOutput : notnull, IDisposable where THidden : notnull, IDisposable
    where TWeightsIn : notnull, IDisposable where TWeightsHidden : notnull, IDisposable where TWeightsOut : notnull, IDisposable
{
    public FfLayer<TInput, THidden, TWeightsIn> InputLayer { get; }
    public List<FfLayer<THidden, THidden, TWeightsHidden>> HiddenLayers { get; }
    public FfLayer<THidden, TOutput, TWeightsOut> OutputLayer { get; }
    
    public string Name { get; set; }
    public Func<Tensor<TOutput>, TOutput, ScalarTensor> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected Func<Tensor<THidden>, Tensor<THidden>> HiddenActivation { get; }
    
    protected FfNetwork(
        FfLayer<TInput, THidden, TWeightsIn> inputLayer,
        List<FfLayer<THidden, THidden, TWeightsHidden>> hiddenLayers,
        FfLayer<THidden, TOutput, TWeightsOut> outputLayer,
        Func<Tensor<TOutput>, TOutput, ScalarTensor> lossFunction,
        int hiddenSize,
        Func<Tensor<THidden>, Tensor<THidden>> hiddenActivation,
        string name = "FfNetwork")
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
        HiddenActivation = hiddenActivation;
        Name = name;
    }

    public TOutput Forward(TInput input)
    {
        THidden hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public Tensor<TOutput> Forward(Tensor<TInput> input)
    {
        Tensor<THidden> hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public void UpdateParameters(ScalarTensorStorage learningRate)
    {
        InputLayer.UpdateParameters(learningRate);
        foreach (var layer in HiddenLayers)
            layer.UpdateParameters(learningRate);
        OutputLayer.UpdateParameters(learningRate);
    }

    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate)
    {
        var lr = Operations.NewDefaultScalarStorage(learningRate);
        NetworkHelpers.Train(
            i => InputLayer.Catalog.ZeroGradient(i), 
            Forward, LossFunction, 
            () => UpdateParameters(lr),
            inputs, targets, epochs);
    }
        

    public double EvaluateLoss(TInput[] inputs, TOutput[] targets) =>
        NetworkHelpers.EvaluateLoss(
            i => InputLayer.Catalog.ZeroGradient(i), 
            Forward, LossFunction, 
            inputs, targets);

    public void Save(string path) => NetworkHelpers.Save(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, path);
    public void Load(string path) => NetworkHelpers.Load(InputLayer, HiddenLayers.Cast<ILayer<THidden,THidden>>().ToList(), OutputLayer, CreateHiddenLayer, path);
    protected abstract FfLayer<THidden, THidden, TWeightsHidden> CreateHiddenLayer();
}