namespace Opal.NNs;

public interface INetwork<in TInput, TOutput>
    where TInput : notnull
    where TOutput : notnull
{
    public string Name { get; }
    
    
    public TOutput Forward(TInput input);
    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate);
    
    public void Save(string path);
    public void Load(string path);
}