using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Rnn;

public class RecurrentLayer<TIn, TOut, TWeights, TBiases, TState> : ILayer<TIn, TOut> 
    where TIn : notnull where TOut : notnull
    where TWeights : notnull
    where TBiases : notnull
    where TState : notnull
{
    public int InputSize { get; }
    public int OutputSize { get; }
    
    public TWeights InputWeights { get; set; }
    public TWeights RecurrentWeights { get; set; }
    public TBiases Biases { get; set; }
    
    public TState HiddenState { get; set; }
    public ActivationFunction<TOut> Activation { get; set; }

    private readonly IRecurrentTensorOperations<TWeights, TBiases, TIn, TOut, TState> tensorOperations;
    private readonly IOptimizer<TWeights, TBiases> optimizer;

    private List<TIn> cachedInputs;
    private List<TState> cachedStates;
    private List<TOut> cachedOutputs;
    private List<TOut> cachedSums;

    public RecurrentLayer(int inputSize, int outputSize, ActivationFunction<TOut> activation, 
        IRecurrentTensorOperations<TWeights, TBiases, TIn, TOut, TState> tensorOperations, IOptimizer<TWeights, TBiases> optimizer)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = activation;
        
        InputWeights = tensorOperations.DefaultWeights(outputSize, inputSize);
        RecurrentWeights = tensorOperations.DefaultWeights(outputSize, outputSize);
        Biases = tensorOperations.DefaultBiases(outputSize);
        HiddenState = tensorOperations.DefaultState(outputSize);
        
        this.tensorOperations = tensorOperations;
        this.optimizer = optimizer;
        cachedInputs = [];
        cachedStates = [];
        cachedOutputs = [];
        cachedSums = [];
    }
    
    public TOut Forward(TIn input, bool cache)
    {
        var inputPart = tensorOperations.Multiply(InputWeights, input);
        var hiddenPart = tensorOperations.Multiply(RecurrentWeights, HiddenState);
        
        var sum = tensorOperations.Add(tensorOperations.Add(inputPart, hiddenPart), Biases);
        var output = Activation.Function(sum);
        HiddenState = tensorOperations.UpdateState(output);

        if (cache)
        {
            cachedInputs.Add(input);
            cachedStates.Add(HiddenState);
            cachedOutputs.Add(output);
            cachedSums.Add(sum);
        }
        
        return output;
    }
    public TOut Forward(TIn input) => Forward(input, true);

    public void Backward(TOut gradOutput, double learningRate)
    {
        var gradInputWeights = tensorOperations.DefaultWeights(OutputSize, InputSize);
        var gradRecurrentWeights = tensorOperations.DefaultWeights(OutputSize, OutputSize);
        var gradBiases = tensorOperations.DefaultBiases(OutputSize);
        TState prevState = HiddenState;

        for (int t = cachedInputs.Count - 1; t >= 0; t--)
        {
            var sum = cachedSums[t];
            var input = cachedInputs[t];
            var state = cachedStates[t];
            var output = cachedOutputs[t];
            
            var gradZ = tensorOperations.Multiply(gradOutput, Activation.Derivative(sum));
            
            gradInputWeights = tensorOperations.Add(gradInputWeights, tensorOperations.GradInputWeights(gradZ, input));
            gradRecurrentWeights = tensorOperations.Add(gradRecurrentWeights, tensorOperations.GradRecurrentWeights(gradZ, prevState));
            gradBiases = tensorOperations.Add(gradBiases, tensorOperations.GradBiases(gradZ));
            
            gradOutput = tensorOperations.GradInput(RecurrentWeights, gradZ);
        }
        
        InputWeights = optimizer.UpdateWeights(InputWeights, gradInputWeights, learningRate);
        RecurrentWeights = optimizer.UpdateWeights(RecurrentWeights, gradRecurrentWeights, learningRate);
        Biases = optimizer.UpdateBiases(Biases, gradBiases, learningRate);
        
        cachedInputs.Clear();
        cachedStates.Clear();
        cachedOutputs.Clear();
        cachedSums.Clear();
    }

    public void Reset()
    {
        HiddenState = tensorOperations.DefaultState(OutputSize);
        InputWeights = tensorOperations.DefaultWeights(OutputSize, InputSize);
        RecurrentWeights = tensorOperations.DefaultWeights(OutputSize, OutputSize);
        Biases = tensorOperations.DefaultBiases(OutputSize);
        cachedInputs.Clear();
        cachedStates.Clear();
        cachedOutputs.Clear();
        cachedSums.Clear();
    }
}

public interface IRecurrentTensorOperations<TWeights, TBiases, TInput, TOutput, TState>
    where TInput : notnull where TOutput : notnull
    where TWeights : notnull
    where TBiases : notnull
    where TState : notnull
{
    public TWeights DefaultWeights(int rows, int cols);
    public TBiases DefaultBiases(int size);
    public TState DefaultState(int size);
    public TInput DefaultInput(int size); 
    // seems weird that we would tell them the dimensions considering we don't know the tensor shape
    // maybe we should switch the ILayer.InputSize and OutputSize requirements to be shapes instead?
    
    public TOutput Add(TOutput a, TBiases b); // this seems weird too- why would we add two tensors of different types?
    public TOutput Add(TOutput a, TOutput b);
    public TWeights Add(TWeights a, TWeights b);
    public TBiases Add(TBiases a, TBiases b);
    public TOutput Multiply(TWeights weights, TInput input);
    public TOutput Multiply(TWeights weights, TState state);
    public TOutput Multiply(TWeights weights, TOutput output);
    public TOutput Multiply(TOutput a, TOutput b);
    public TWeights GradInputWeights(TOutput gradZ, TInput input);
    public TWeights GradRecurrentWeights(TOutput gradZ, TState state);
    public TBiases GradBiases(TOutput gradZ);
    public TOutput GradInput(TWeights weights, TOutput gradZ);
    
    
    public TState UpdateState(TOutput output);
}