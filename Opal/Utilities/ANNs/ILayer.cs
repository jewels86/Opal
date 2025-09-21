namespace Opal.Utilities.ANNs;

public interface ILayer<T> where T : notnull
{
    public int InputSize { get; }
    public int OutputSize { get; }
    
    public T Forward(T input);
    public T Backward(T gradOutput, double learningRate);
    public void Reset();
}