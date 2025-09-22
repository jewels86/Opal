namespace Opal.Utilities.ANNs;

public interface ILayer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public int InputSize { get; }
    public int OutputSize { get; }
    
    public TOutput Forward(TInput input);
    public void Backward(TOutput gradOutput, double learningRate);
    public void Reset();
}