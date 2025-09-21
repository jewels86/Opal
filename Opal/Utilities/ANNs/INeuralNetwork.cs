namespace Opal.Utilities.ANNs;

public interface INeuralNetwork<in TInput, TOutput>
    where TInput : notnull
    where TOutput : notnull
{
    public TOutput Forward(TInput input);
    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate);
    public double EvaluateLoss(TInput[] inputs, TOutput[] targets);
    public void Reset();
}