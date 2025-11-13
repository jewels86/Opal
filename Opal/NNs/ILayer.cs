namespace Opal.NNs;

public interface ILayer<in TIn, out TOut>
    where TIn : notnull where TOut : notnull
{
    public TOut Forward(TIn input);
    
    public void Write(BinaryWriter writer);
    public void Read(BinaryReader reader);
}