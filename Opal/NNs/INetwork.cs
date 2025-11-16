namespace Opal.NNs;

public interface INetwork<in TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    public TOut Forward(TIn input);
    public void Train(TIn[] inputs, TOut[] targets, int epochs, double learningRate);
    
    public void Save(string path);
    public void Load(string path);
}