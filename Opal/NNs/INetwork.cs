using Jewels.Lazulite;

namespace Opal.NNs;

public interface INetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    public Tensor<TOut> Forward(Tensor<TIn> input);
    public Value<TOut> Forward(Value<TIn> input);
    
    public void Train(Value<TIn>[] inputs, Value<TOut>[] targets, int epochs, float lr);
    
    public void Save(string path);
    public void Load(string path);
}