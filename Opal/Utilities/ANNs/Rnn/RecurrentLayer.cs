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
    
    private IRecurrentTensorOperations<TWeights, TBiases, TIn, TOut, TState> TensorOperations { get; }
    private IOptimizer<TWeights, TBiases> Optimizer { get; }
    
    private TIn lastInput;

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
        
        lastInput = tensorOperations.DefaultInput(inputSize);
        TensorOperations = tensorOperations;
        Optimizer = optimizer;
    }
    
    public TOut Forward(TIn input)
    {
        var inputPart = TensorOperations.Multiply(InputWeights, input);
        var hiddenPart = TensorOperations.Multiply(RecurrentWeights, HiddenState);
        
        var sum = TensorOperations.Add(TensorOperations.Add(inputPart, hiddenPart), Biases);
        var output = Activation.Function(sum);
        HiddenState = TensorOperations.UpdateState(output);
        return output;
    }

    public TIn Backward(TOut gradOutput, double learningRate)
    {
        var gradZ = Activation.Derivative(gradOutput);
        
        var gradInputWeights = TensorOperations.GradInputWeights(gradZ, lastInput);
        var gradRecurrentWeights = TensorOperations.GradRecurrentWeights(gradZ, HiddenState);
        var gradBiases = TensorOperations.GradBiases(gradZ);
        
        InputWeights = Optimizer.UpdateWeights(InputWeights, gradInputWeights, learningRate);
        RecurrentWeights = Optimizer.UpdateWeights(RecurrentWeights, gradRecurrentWeights, learningRate);
        Biases = Optimizer.UpdateBiases(Biases, gradBiases, learningRate);
        
        var gradInput = TensorOperations.GradInput(InputWeights, gradZ);
        return gradInput;
    }

    public void Reset()
    {
        HiddenState = TensorOperations.DefaultState(OutputSize);
        lastInput = TensorOperations.DefaultInput(InputSize);
        InputWeights = TensorOperations.DefaultWeights(OutputSize, InputSize);
        RecurrentWeights = TensorOperations.DefaultWeights(OutputSize, OutputSize);
        Biases = TensorOperations.DefaultBiases(OutputSize);
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
    
    public TOutput Add(TOutput a, TBiases b);
    public TOutput Add(TOutput a, TOutput b);
    public TOutput Multiply(TWeights weights, TInput input);
    public TOutput Multiply(TWeights weights, TState state);
    public TWeights GradInputWeights(TOutput gradZ, TInput input);
    public TWeights GradRecurrentWeights(TOutput gradZ, TState state);
    public TBiases GradBiases(TOutput gradZ);
    public TInput GradInput(TWeights weights, TOutput gradZ);
    
    public TState UpdateState(TOutput output);
}