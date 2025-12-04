using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public interface INetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    public Tensor<TOut> Forward(Tensor<TIn> input);
    public Value<TOut> Forward(Value<TIn> input);
    
    public List<float> Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr);
    
    public void Save(string path);
    public void Load(string path);
}

public interface ISequentialNetwork<TIn, TOut> where TIn : notnull where TOut : notnull 
{
    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] input);
    public Value<TOut> ForwardSequence(Value<TIn>[] sequence);
    
    public void TrainSequences(Value<TIn>[][] sequences, Value<TOut>[] targets, int epochs, float lr);
    public float EvaluateLossSequences(Value<TIn>[][] sequences, Value<TOut>[] targets);
}

public interface ITransformingNetwork<TIn, TOut> where TIn : notnull where TOut : notnull
{
    public Tensor<TOut>[] ForwardTransforming(Tensor<TIn>[] input);
    public Value<TOut>[] ForwardTransforming(Value<TIn>[] sequence);
    
    public void TrainTransforming(Value<TIn>[][] sequences, Value<TOut>[][] targets, int epochs, double learningRate);
    public double EvaluateLossTransforming(Value<TIn>[][] sequences, Value<TOut>[][] targets);
}