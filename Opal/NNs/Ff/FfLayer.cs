using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class FfLayer<TWeights, TBiases, TInput, TOutput> : ILayer<TInput, TOutput> 
    where TInput : notnull where TOutput : notnull
{
    public TWeights Weights { get; set; }
    public TBiases Biases { get; set; }
    public int[] InputShape { get; }
    public int[] OutputShape { get; }
    public ActivationFunction<TOutput> Activation { get; set; }
    
    private IFfTensorOperations<TWeights, TBiases, TInput, TOutput> TensorOperations { get; }
    private IOptimizer<TWeights, TBiases> Optimizer { get; }

    private TInput lastInput;
    private TOutput lastZ;

    public FfLayer(int[] inputShape, int[] outputShape, ActivationFunction<TOutput> activation, 
        IFfTensorOperations<TWeights, TBiases, TInput, TOutput> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer)
    {
        InputShape = inputShape;
        OutputShape = outputShape;
        Activation = activation;
        
        Weights = tensorOperations.DefaultWeights(outputShape, inputShape);
        Biases = tensorOperations.DefaultBiases(outputShape);
        lastInput = tensorOperations.DefaultInput(inputShape);
        lastZ = tensorOperations.DefaultOutput(outputShape);
        
        TensorOperations = tensorOperations;
        Optimizer = optimizer;
    }

    public TOutput Forward(TInput input)
    {
        var z = TensorOperations.Add(TensorOperations.Multiply(Weights, input), Biases);
        lastInput = input;
        lastZ = z;
        return Activation.Function(z);
    }

    public TInput Backward(TOutput gradOutput, double learningRate)
    {
        var gradZ = TensorOperations.Multiply(gradOutput, Activation.Derivative(lastZ));
        var gradWeights = TensorOperations.GradWeights(gradZ, lastInput);
        var gradBiases = TensorOperations.GradBiases(gradZ);
        var gradInput = TensorOperations.GradInput(Weights, gradZ);

        Weights = Optimizer.UpdateWeights(Weights, gradWeights, learningRate);
        Biases = Optimizer.UpdateBiases(Biases, gradBiases, learningRate);

        return gradInput;
    }
    public void Reset()
    {
        Weights = TensorOperations.DefaultWeights(OutputShape, InputShape);
        Biases = TensorOperations.DefaultBiases(OutputShape);
        lastInput = TensorOperations.DefaultInput(InputShape);
        lastZ = TensorOperations.DefaultOutput(OutputShape);
    }
}

public interface IFfTensorOperations<TWeights, TBiases, TInput, TOutput>
{
    TOutput Multiply(TWeights weights, TInput input);
    TOutput Multiply(TOutput a, TOutput b);
    TOutput Add(TOutput output, TBiases biases);
    TOutput Apply(TOutput output, Func<double, double> activation);

    TInput GradInput(TWeights weights, TOutput gradZ);
    TWeights GradWeights(TOutput gradZ, TInput lastInput);
    TBiases GradBiases(TOutput gradZ);

    TInput DefaultInput(int[] inputShape);
    TOutput DefaultOutput(int[] outputsShape);
    TWeights DefaultWeights(int[] outputShape, int[] inputShape);
    TBiases DefaultBiases(int[] outputShape);
}