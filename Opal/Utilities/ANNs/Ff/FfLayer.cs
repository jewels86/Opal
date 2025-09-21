using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class FfLayer<TWeights, TBiases, TInput, TOutput> : ILayer<TInput, TOutput> 
    where TInput : notnull where TOutput : notnull
{
    public TWeights Weights { get; set; }
    public TBiases Biases { get; set; }
    public int InputSize { get; }
    public int OutputSize { get; }
    public ActivationFunction<TOutput> Activation { get; set; }
    
    private IFfTensorOperations<TWeights, TBiases, TInput, TOutput> TensorOperations { get; }
    private IFfOptimizer<TWeights, TBiases> Optimizer { get; }

    private TInput lastInput;

    public FfLayer(int inputSize, int outputSize, ActivationFunction<TOutput> activation, 
        IFfTensorOperations<TWeights, TBiases, TInput, TOutput> tensorOperations,
        IFfOptimizer<TWeights, TBiases> optimizer)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = activation;
        
        Weights = tensorOperations.DefaultWeights(outputSize, inputSize);
        Biases = tensorOperations.DefaultBiases(outputSize);
        lastInput = tensorOperations.DefaultInput(inputSize);
        
        TensorOperations = tensorOperations;
        Optimizer = optimizer;
    }

    public TOutput Forward(TInput input)
    {
        var z = TensorOperations.Add(TensorOperations.Multiply(Weights, input), Biases);
        return Activation.Function(z);
    }

    public TInput Backward(TOutput gradOutput, double learningRate)
    {
        var gradZ = Activation.Derivative(gradOutput);
        var gradWeights = TensorOperations.GradWeights(gradZ, lastInput);
        var gradBiases = TensorOperations.GradBiases(gradZ);
        var gradInput = TensorOperations.GradInput(Weights, gradZ);

        Weights = Optimizer.UpdateWeights(Weights, gradWeights, learningRate);
        Biases = Optimizer.UpdateBiases(Biases, gradBiases, learningRate);

        return gradInput;
    }
    
    public void Reset()
    {
        Weights = TensorOperations.DefaultWeights(OutputSize, InputSize);
        Biases = TensorOperations.DefaultBiases(OutputSize);
        lastInput = TensorOperations.DefaultInput(InputSize);
    }
}

public interface IFfTensorOperations<TWeights, TBiases, TInput, TOutput>
{
    TOutput Multiply(TWeights weights, TInput input);
    TOutput Add(TOutput output, TBiases biases);
    TOutput Apply(TOutput output, Func<double, double> activation);

    TInput GradInput(TWeights weights, TOutput gradZ);
    TWeights GradWeights(TOutput gradZ, TInput lastInput);
    TBiases GradBiases(TOutput gradZ);

    TInput DefaultInput(int size);
    TOutput DefaultOutput(int size);
    TWeights DefaultWeights(int rows, int cols);
    TBiases DefaultBiases(int size);
}

public interface IFfOptimizer<TWeights, TBiases>
{
    TWeights UpdateWeights(TWeights weights, TWeights gradWeights, double learningRate);
    TBiases UpdateBiases(TBiases biases, TBiases gradBiases, double learningRate);
}
