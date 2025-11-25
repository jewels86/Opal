using Jewels.Lazulite;

namespace Opal.NNs;

public interface ISequentialNetwork<TIn, TOut> : INetwork<TIn, TOut> where TIn : notnull where TOut : notnull 
{
    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] input);
    public Value<TOut> ForwardSequence(Value<TIn>[] sequence);
    
    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr);
    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets);
}