using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Rnn;

public class RecurrentLayer<TWeights, TBiases, TState, TIn, TOut> : ILayer<TIn, TOut> 
    where TIn : notnull where TOut : notnull
    where TWeights : notnull
    where TBiases : notnull
    where TState : notnull
{
    public int[] InputShape { get; }
    public int[] OutputShape { get; }
    
    public TWeights InputWeights { get; set; }
    public TWeights RecurrentWeights { get; set; }
    public TBiases Biases { get; set; }
    
    public TState HiddenState { get; set; }
    public ActivationFunction<TOut> Activation { get; }

    private readonly IRecurrentTensorOperations<TWeights, TBiases, TIn, TOut, TState> tensorOperations;
    private readonly IOptimizer<TWeights, TBiases> optimizer;

    private List<TIn> cachedInputs;
    private List<TState> cachedStates;
    private List<TOut> cachedOutputs;
    private List<TOut> cachedSums;

    public RecurrentLayer(int[] inputShape, int[] outputShape, ActivationFunction<TOut> activation, 
        IRecurrentTensorOperations<TWeights, TBiases, TIn, TOut, TState> tensorOperations, IOptimizer<TWeights, TBiases> optimizer)
    {
        InputShape = inputShape;
        OutputShape = outputShape;
        Activation = activation;
        
        InputWeights = tensorOperations.DefaultWeights(outputShape, inputShape);
        RecurrentWeights = tensorOperations.DefaultWeights(outputShape, outputShape);
        Biases = tensorOperations.DefaultBiases(outputShape);
        HiddenState = tensorOperations.DefaultState(outputShape);
        
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

    public TIn Backward(TOut gradOutput, double learningRate)
    {
        var gradInputWeights = tensorOperations.DefaultWeights(OutputShape, InputShape);
        var gradRecurrentWeights = tensorOperations.DefaultWeights(OutputShape, OutputShape);
        var gradBiases = tensorOperations.DefaultBiases(OutputShape);
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
            
            gradOutput = tensorOperations.GradOutput(RecurrentWeights, gradZ);
            prevState = state;
        }
        
        InputWeights = optimizer.UpdateWeights(InputWeights, gradInputWeights, learningRate);
        RecurrentWeights = optimizer.UpdateWeights(RecurrentWeights, gradRecurrentWeights, learningRate);
        Biases = optimizer.UpdateBiases(Biases, gradBiases, learningRate);
        
        cachedInputs.Clear();
        cachedStates.Clear();
        cachedOutputs.Clear();
        cachedSums.Clear();
        
        return tensorOperations.GradInput(InputWeights, gradOutput);
    }

    public void Reset()
    {
        HiddenState = tensorOperations.DefaultState(OutputShape);
        InputWeights = tensorOperations.DefaultWeights(OutputShape, InputShape);
        RecurrentWeights = tensorOperations.DefaultWeights(OutputShape, OutputShape);
        Biases = tensorOperations.DefaultBiases(OutputShape);
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
    public TWeights DefaultWeights(int[] outputShape, int[] inputShape);
    public TBiases DefaultBiases(int[] outputShape);
    public TState DefaultState(int[] outputsShape);
    
    public TOutput Add(TOutput a, TBiases b);
    public TOutput Add(TOutput a, TOutput b);
    public TWeights Add(TWeights a, TWeights b);
    public TBiases Add(TBiases a, TBiases b);
    public TOutput Multiply(TWeights weights, TInput input);
    public TOutput Multiply(TWeights weights, TState state);
    public TOutput Multiply(TOutput a, TOutput b);
    public TWeights GradInputWeights(TOutput gradZ, TInput input);
    public TWeights GradRecurrentWeights(TOutput gradZ, TState state);
    public TBiases GradBiases(TOutput gradZ);
    public TOutput GradOutput(TWeights weights, TOutput gradZ);
    public TInput GradInput(TWeights weights, TOutput gradZ);
    
    
    public TState UpdateState(TOutput output);
}