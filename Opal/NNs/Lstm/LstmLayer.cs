using Opal.Autograd;
using Opal.Mathematics;

namespace Opal.NNs.Lstm;

public class LstmLayer<TIn, TOut, TWeight> : ILayer<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeight : notnull
{
    public Tensor<TWeight>[] EncoderForgetWeights { get; set; } 
    public Tensor<TWeight>[] EncoderInputWeights { get; set; } 
    public Tensor<TWeight>[] EncoderCellWeights { get; set; } 
    public Tensor<TWeight>[] EncoderOutputWeights { get; set; }
    public Tensor<TOut> EncoderForgetBiases { get; set; }
    public Tensor<TOut> EncoderInputBiases { get; set; }
    public Tensor<TOut> EncoderCellBiases { get; set; }
    public Tensor<TOut> EncoderOutputBiases { get; set; }
    
    public Tensor<TWeight>[] DecoderForgetWeights { get; set; } 
    public Tensor<TWeight>[] DecoderInputWeights { get; set; } 
    public Tensor<TWeight>[] DecoderCellWeights { get; set; } 
    public Tensor<TWeight>[] DecoderOutputWeights { get; set; }
    
    public Tensor<TOut> DecoderForgetBiases { get; set; }
    public Tensor<TOut> DecoderInputBiases { get; set; }
    public Tensor<TOut> DecoderCellBiases { get; set; }
    public Tensor<TOut> DecoderOutputBiases { get; set; }
    
    public ActivationFunction<TOut> SigmoidActivation { get; set; }
    public ActivationFunction<TOut> TanhActivation { get; set; }
    public ILstmCatalog<TIn, TOut, TWeight> Catalog { get; set; }

    public (Tensor<TOut> hidden, Tensor<TOut> state) EncoderForward(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatInputHidden(input, prevHidden);
        Tensor<TOut> forgetGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyEncoderWeights(concat, EncoderForgetWeights), EncoderForgetBiases));
        Tensor<TOut> inputGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyEncoderWeights(concat, EncoderInputWeights), EncoderInputBiases));
        Tensor<TOut> cellGate = TanhActivation.Function(Catalog.AddOut(Catalog.MultiplyEncoderWeights(concat, EncoderCellWeights), EncoderCellBiases));
        Tensor<TOut> outputGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyEncoderWeights(concat, EncoderOutputWeights), EncoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.AddOut(Catalog.MultiplyOut(forgetGate, state), Catalog.MultiplyOut(inputGate, cellGate));
        Tensor<TOut> newHidden = Catalog.MultiplyOut(outputGate, TanhActivation.Function(newState));
        
        return (newHidden, newState);
    }
    
    public (Tensor<TOut> hidden, Tensor<TOut> state) DecoderForward(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatHidden(input, prevHidden);
        Tensor<TOut> forgetGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyDecoderWeights(concat, DecoderForgetWeights), DecoderForgetBiases));
        Tensor<TOut> inputGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyDecoderWeights(concat, DecoderInputWeights), DecoderInputBiases));
        Tensor<TOut> cellGate = TanhActivation.Function(Catalog.AddOut(Catalog.MultiplyDecoderWeights(concat, DecoderCellWeights), DecoderCellBiases));
        Tensor<TOut> outputGate = SigmoidActivation.Function(Catalog.AddOut(Catalog.MultiplyDecoderWeights(concat, DecoderOutputWeights), DecoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.AddOut(Catalog.MultiplyOut(forgetGate, state), Catalog.MultiplyOut(inputGate, cellGate));
        Tensor<TOut> newHidden = Catalog.MultiplyOut(outputGate, TanhActivation.Function(newState));
        
        return (newHidden, newState);
    }

    public Tensor<TOut> Forward(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        Tensor<TOut> encoderOutputs = EncoderForward(input, initialState, initialHidden).hidden;
        Tensor<TOut> decoderOutputs = DecoderForward(encoderOutputs, initialState, initialHidden).hidden;
        return decoderOutputs;
    }
}

public interface ILstmCatalog<TIn, TOut, TWeight>
    where TIn : notnull where TOut : notnull
    where TWeight : notnull
{
    Tensor<TOut> ConcatInputHidden(Tensor<TIn> input, Tensor<TOut> prevHidden);
    Tensor<TOut> ConcatHidden(Tensor<TOut> input, Tensor<TOut> prevHidden);
    Tensor<TOut> AddOut(Tensor<TOut> a, Tensor<TOut> b);
    Tensor<TOut> MultiplyEncoderWeights(Tensor<TOut> concat, Tensor<TWeight>[] weights);
    Tensor<TOut> MultiplyDecoderWeights(Tensor<TOut> concat, Tensor<TWeight>[] weights);
    public Tensor<TOut> MultiplyOut(Tensor<TOut> a, Tensor<TOut> b);
}