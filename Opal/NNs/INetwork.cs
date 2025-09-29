namespace Opal.NNs;

public interface INetwork<in TInput, TOutput>
    where TInput : notnull
    where TOutput : notnull
{
    public string Name { get; }
    public int[] InputShape { get; }
    public int[] OutputShape { get; }
    
    
    public TOutput Forward(TInput input);
    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate);
    public double EvaluateLoss(TInput[] inputs, TOutput[] targets);
    public void Reset();
}