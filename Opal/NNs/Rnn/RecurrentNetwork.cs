using Opal.Mathematics;

namespace Opal.NNs.Rnn;

public class RecurrentNetwork<TWeights, TBiases, TState, TInput, THidden, TOutput> : INetwork<TInput, TOutput>
    where TInput : notnull where TOutput : notnull
    where THidden : notnull
    where TWeights : notnull
    where TBiases : notnull
    where TState : notnull

{
    public RecurrentLayer<TWeights, TBiases, TState, TInput, THidden> InputLayer { get; }
    public List<RecurrentLayer<TWeights, TBiases, TState, THidden, THidden>> HiddenLayers { get; }
    public RecurrentLayer<TWeights, TBiases, TState, THidden, TOutput> OutputLayer { get; }
    
    public string Name { get; }
    
    public int[] InputShape { get; }
    public int[] HiddenShape { get; }
    public int[] OutputShape { get; }
    
    public IRecurrentTensorOperations<TWeights, TBiases, TInput, THidden, TState> InputTensorOperations { get; }
    public IRecurrentTensorOperations<TWeights, TBiases, THidden, THidden, TState> HiddenTensorOperations { get; }
    public IRecurrentTensorOperations<TWeights, TBiases, THidden, TOutput, TState> OutputTensorOperations { get; }
    
    public ActivationFunction<TOutput> OutputActivation { get; }
    public ActivationFunction<THidden> HiddenActivation { get; }
    
    public LossFunction<TOutput> LossFunction { get; }
    public IOptimizer<TWeights, TBiases> Optimizer { get; }
    
    public RecurrentNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<THidden> hiddenActivation, ActivationFunction<TOutput> outputActivation,
        LossFunction<TOutput> lossFunction, IOptimizer<TWeights, TBiases> optimizer, 
        IRecurrentTensorOperations<TWeights, TBiases, TInput, THidden, TState> inputTensorOperations,
        IRecurrentTensorOperations<TWeights, TBiases, THidden, THidden, TState> hiddenTensorOperations,
        IRecurrentTensorOperations<TWeights, TBiases, THidden, TOutput, TState> outputTensorOperations,
        string name = "RecurrentNetwork")
    {
        InputShape = inputShape;
        HiddenShape = hiddenShape;
        OutputShape = outputShape;
        Name = name;
        
        InputTensorOperations = inputTensorOperations;
        HiddenTensorOperations = hiddenTensorOperations;
        OutputTensorOperations = outputTensorOperations;

        OutputActivation = outputActivation;
        HiddenActivation = hiddenActivation;
        LossFunction = lossFunction;
        Optimizer = optimizer;
        
        InputLayer = new(InputShape, HiddenShape, HiddenActivation, InputTensorOperations, Optimizer);
        HiddenLayers = [];
        for (int i = 0; i < hiddenLayers; i++)
            HiddenLayers.Add(new(HiddenShape, HiddenShape, HiddenActivation, HiddenTensorOperations, Optimizer));
        OutputLayer = new(HiddenShape, OutputShape, OutputActivation, OutputTensorOperations, Optimizer);
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