using Jewels.Lazulite;

namespace Opal.NNs;

public interface ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull
{
    public Tensor<TOut> Forward(Tensor<TIn> input);
    public Value<TOut> Forward(Value<TIn> input);
    
    public void ZeroGradients();
    
    public void Write(BinaryWriter writer);
    public void Read(BinaryReader reader);
}