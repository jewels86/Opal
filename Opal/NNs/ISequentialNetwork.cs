namespace Opal.NNs;

public interface ISequentialNetwork<in TIn, TOut>
{
    public TOut ForwardSequence(TIn[] sequence);
    public void TrainSequences(TIn[][] sequences, TOut[] targets, int epochs, double learningRate);
    public double EvaluateLossSequences(TIn[][] sequences, TOut[] targets);
}