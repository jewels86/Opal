namespace Opal.NNs;

public interface ISequentialNetwork<TIn, TOut> : INetwork<TIn, TOut> where TIn : notnull where TOut : notnull 
{
    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] input);
    public TOut ForwardSequence(TIn[] sequence);
    
    public void TrainSequences(TIn[][] sequences, TOut[] targets, int epochs, double learningRate);
    public double EvaluateLossSequences(TIn[][] sequences, TOut[] targets);
}