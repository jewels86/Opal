using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public abstract class FfNetwork<TWeights, TBiases, TInput, THidden, TOutput> : INeuralNetwork<TInput, TOutput>
    where TInput : notnull where TOutput : notnull
    where THidden : notnull
    where TWeights : notnull where TBiases : notnull
{
    public FfLayer<TWeights, TBiases, TInput, THidden> InputLayer { get; }
    public List<FfLayer<TWeights, TBiases, THidden, THidden>> HiddenLayers { get; }
    public FfLayer<TWeights, TBiases, THidden, TOutput> OutputLayer { get; }
    
    public string Name { get; }
    
    public int InputSize { get; }
    public int HiddenSize { get; }
    public int OutputSize { get; }
    
    public IFfTensorOperations<TWeights, TBiases, TInput, THidden> InputTensorOperations { get; }
    public IFfTensorOperations<TWeights, TBiases, THidden, THidden> HiddenTensorOperations { get; }
    public IFfTensorOperations<TWeights, TBiases, THidden, TOutput> OutputTensorOperations { get; }
    
    public ActivationFunction<TOutput> OutputActivation { get; }
    public ActivationFunction<THidden> HiddenActivation { get; }
    
    public LossFunction<TOutput> LossFunction { get; }
    public IFfOptimizer<TWeights, TBiases> Optimizer { get; }
    
    protected FfNetwork(int inputSize, int hiddenSize, int outputSize, int hiddenLayers,
        ActivationFunction<THidden> hiddenActivation, ActivationFunction<TOutput> outputActivation,
        LossFunction<TOutput> lossFunction, IFfOptimizer<TWeights, TBiases> optimizer, 
        IFfTensorOperations<TWeights, TBiases, TInput, THidden> inputTensorOperations,
        IFfTensorOperations<TWeights, TBiases, THidden, THidden> hiddenTensorOperations,
        IFfTensorOperations<TWeights, TBiases, THidden, TOutput> outputTensorOperations,
        string name = "FfNetwork")
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        OutputSize = outputSize;
        Name = name;
        
        InputTensorOperations = inputTensorOperations;
        HiddenTensorOperations = hiddenTensorOperations;
        OutputTensorOperations = outputTensorOperations;

        OutputActivation = outputActivation;
        HiddenActivation = hiddenActivation;
        LossFunction = lossFunction;
        Optimizer = optimizer;
        
        InputLayer = new(InputSize, HiddenSize, HiddenActivation, InputTensorOperations, Optimizer);
        HiddenLayers = [];
        for (int i = 0; i < hiddenLayers; i++)
            HiddenLayers.Add(new(HiddenSize, HiddenSize, HiddenActivation, HiddenTensorOperations, Optimizer));
        OutputLayer = new(HiddenSize, OutputSize, OutputActivation, OutputTensorOperations, Optimizer);
    }
    

    public TOutput Forward(TInput input)
    {
        THidden hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var target = targets[i];
                var hidden = InputLayer.Forward(input);
                foreach (var layer in HiddenLayers)
                    hidden = layer.Forward(hidden);
                var output = OutputLayer.Forward(hidden);

                var lossGrad = LossFunction.Derivative(output, target);

                var grad = OutputLayer.Backward(lossGrad, learningRate);
                for (int h = HiddenLayers.Count - 1; h >= 0; h--)
                    grad = HiddenLayers[h].Backward(grad, learningRate);
                InputLayer.Backward(grad, learningRate);
            }
        }
    }

    public double EvaluateLoss(TInput[] inputs, TOutput[] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var predicted = Forward(inputs[i]);
            var actual = targets[i];
            totalLoss += LossFunction.Function(predicted, actual);
        }
        return totalLoss / inputs.Length;
    }
    public void Reset()
    {
        InputLayer.Reset();
        foreach (var layer in HiddenLayers)
            layer.Reset();
        OutputLayer.Reset();
    }
}