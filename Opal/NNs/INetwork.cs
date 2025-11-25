namespace Opal.NNs;

public interface INetwork<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    public Tensor<TOut> Forward(Tensor<TIn> input);
    public TOut Forward(TIn input);
    
    public void Train(TIn[] inputs, TOut[] targets, int epochs, double learningRate);
    
    public void Save(string path);
    public void Load(string path);
}