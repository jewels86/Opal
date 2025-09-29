namespace Opal.NNs;

public interface ILayer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public int[] InputShape { get; }
    public int[] OutputShape { get; }
    
    public TOutput Forward(TInput input);
    public TInput Backward(TOutput gradOutput, double learningRate);
    public void Reset();
}