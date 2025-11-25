namespace Opal.NNs;

public interface ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull
{
    public Tensor<TOut> Forward(Tensor<TIn> input);
    public TOut Forward(TIn input);
    
    public void Write(BinaryWriter writer);
    public void Read(BinaryReader reader);
}