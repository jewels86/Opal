using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public class LstmNetwork<TIn, THidden, TOut, TWeightIn, TWeightHidden, TWeightOut> : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where THidden : notnull
    where TOut : notnull
    where TWeightIn : notnull
    where TWeightHidden : notnull
    where TWeightOut : notnull
{
    public LstmLayer<TIn, THidden, TWeightIn> InputLayer { get; set; }
    public List<LstmLayer<THidden, THidden, TWeightHidden>> HiddenLayers { get; set; }
    public LstmLayer<THidden, TOut, TWeightOut> OutputLayer { get; set; }
    
    public string Name { get; }
    public LossFunction<TOut> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected ActivationFunction<THidden> HiddenActivation { get; }
    
    
}
