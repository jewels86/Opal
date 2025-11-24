using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public class LstmLayer<TIn, TOut, TWeights> : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public required Tensor<TWeights> EncoderForgetWeights { get; set; } 
    public required Tensor<TWeights> EncoderInputWeights { get; set; } 
    public required Tensor<TWeights> EncoderCellWeights { get; set; } 
    public required Tensor<TWeights> EncoderOutputWeights { get; set; }
    public required Tensor<TOut> EncoderForgetBiases { get; set; }
    public required Tensor<TOut> EncoderInputBiases { get; set; }
    public required Tensor<TOut> EncoderCellBiases { get; set; }
    public required Tensor<TOut> EncoderOutputBiases { get; set; }
    
    public required Tensor<TWeights> DecoderForgetWeights { get; set; } 
    public required Tensor<TWeights> DecoderInputWeights { get; set; } 
    public required Tensor<TWeights> DecoderCellWeights { get; set; } 
    public required Tensor<TWeights> DecoderOutputWeights { get; set; }
    
    public required Tensor<TOut> DecoderForgetBiases { get; set; }
    public required Tensor<TOut> DecoderInputBiases { get; set; }
    public required Tensor<TOut> DecoderCellBiases { get; set; }
    public required Tensor<TOut> DecoderOutputBiases { get; set; }
    
    public required Func<Tensor<TOut>, Tensor<TOut>> SigmoidActivation { get; set; }
    public required Func<Tensor<TOut>, Tensor<TOut>> TanhActivation { get; set; }
    public required ILstmCatalog<TIn, TOut, TWeights> Catalog { get; set; }
    
    public required Tensor<TOut> DefaultState { get; set; }
    public required Tensor<TOut> DefaultHidden { get; set; }

    #region Encoder/Decoder
    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Encoder(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        using Tensor<TOut> concat = Catalog.ConcatInputHidden(input, prevHidden);
        using Tensor<TOut> forgetGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(EncoderForgetWeights, concat).Defer(), EncoderForgetBiases).Defer());
        using Tensor<TOut> inputGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(EncoderInputWeights, concat).Defer(), EncoderInputBiases).Defer());
        using Tensor<TOut> cellGate = TanhActivation(Catalog.Add(Catalog.Multiply(EncoderCellWeights, concat).Defer(), EncoderCellBiases).Defer());
        using Tensor<TOut> outputGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(EncoderOutputWeights, concat).Defer(), EncoderOutputBiases).Defer());
        
        using Tensor<TOut> newState = Catalog.Add(Catalog.Multiply(forgetGate, state).Defer(), Catalog.Multiply(inputGate, cellGate).Defer());
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, TanhActivation(newState).Defer());
        
        return (newHidden, newState);
    }
    
    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Decoder(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatHidden(input, prevHidden);
        Tensor<TOut> forgetGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(DecoderForgetWeights, concat).Defer(), DecoderForgetBiases).Defer());
        Tensor<TOut> inputGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(DecoderInputWeights, concat).Defer(), DecoderInputBiases).Defer());
        Tensor<TOut> cellGate = TanhActivation(Catalog.Add(Catalog.Multiply(DecoderCellWeights, concat).Defer(), DecoderCellBiases).Defer());
        Tensor<TOut> outputGate = SigmoidActivation(Catalog.Add(Catalog.Multiply(DecoderOutputWeights, concat).Defer(), DecoderOutputBiases).Defer());

        
        Tensor<TOut> newState = Catalog.Add(Catalog.Multiply(forgetGate, state).Defer(), Catalog.Multiply(inputGate, cellGate).Defer());
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, TanhActivation(newState).Defer());
        
        return (newHidden, newState);
    }

    public Tensor<TOut>[] EncoderSequence(Tensor<TIn>[] inputs, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        List<Tensor<TOut>> hiddenStates = new();
        Tensor<TOut> currentHidden = initialHidden;
        Tensor<TOut> currentState = initialState;
        
        foreach (var input in inputs)
        {
            var (hidden, state) = Encoder(input, currentState, currentHidden);
            hiddenStates.Add(hidden);
            currentHidden = hidden;
            currentState = state;
        }
        
        return hiddenStates.ToArray();
    }
    
    public Tensor<TOut>[] DecoderSequence(Tensor<TOut>[] inputs, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        List<Tensor<TOut>> hiddenStates = new();
        Tensor<TOut> currentHidden = initialHidden;
        Tensor<TOut> currentState = initialState;
        
        foreach (var input in inputs)
        {
            var (hidden, state) = Decoder(input, currentState, currentHidden);
            hiddenStates.Add(hidden);
            currentHidden = hidden;
            currentState = state;
        }
        
        return hiddenStates.ToArray();
    }
    #endregion

    public virtual Tensor<TOut> ForwardCore(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        var encoderOutput = Encoder(input, initialState, initialHidden);
        var decoderOutput = Decoder(encoderOutput.hidden, encoderOutput.state, encoderOutput.hidden);
        return decoderOutput.hidden;
    }
    
    #region Overloads
    public TOut Forward(TIn input)
    {
        Tensor<TIn> tensorInput = new(input, null, _ => { }, Catalog.ZeroGradient(input));
        Tensor<TOut> initialHidden = DefaultHidden;
        Tensor<TOut> initialState = DefaultState;
        
        return ForwardCore(tensorInput, initialHidden, initialState).Value;
    }
    
    public Tensor<TOut> Forward(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState) => ForwardCore(input, initialHidden, initialState);
    public Tensor<TOut> Forward(Tensor<TIn> input) => ForwardCore(input, DefaultHidden, DefaultState);

    public TOut ForwardSequence(TIn[] inputs)
    {
        Tensor<TOut> initialHidden = DefaultHidden;
        Tensor<TOut> initialState = DefaultState;
        
        var tensorInputs = inputs.Select(i => new Tensor<TIn>(i, null, _ => { }, Catalog.ZeroGradient(i))).ToArray();
        var encoderOutputs = EncoderSequence(tensorInputs, initialHidden, initialState);
        var decoderOutputs = DecoderSequence(encoderOutputs, initialHidden, initialState);
        
        return decoderOutputs[^1].Value;
    }
    #endregion

    public virtual void UpdateParameters(double lr)
    {
        EncoderForgetWeights.Value = Catalog.Subtract(EncoderForgetWeights.Value, Catalog.Scale(EncoderForgetWeights.Gradient, lr));
        EncoderForgetWeights.Gradient = Catalog.ZeroGradient(EncoderForgetWeights.Value);
        EncoderInputWeights.Value = Catalog.Subtract(EncoderInputWeights.Value, Catalog.Scale(EncoderInputWeights.Gradient, lr));
        EncoderInputWeights.Gradient = Catalog.ZeroGradient(EncoderInputWeights.Value);
        EncoderCellWeights.Value = Catalog.Subtract(EncoderCellWeights.Value, Catalog.Scale(EncoderCellWeights.Gradient, lr));
        EncoderCellWeights.Gradient = Catalog.ZeroGradient(EncoderCellWeights.Value);
        EncoderOutputWeights.Value = Catalog.Subtract(EncoderOutputWeights.Value, Catalog.Scale(EncoderOutputWeights.Gradient, lr));
        EncoderOutputWeights.Gradient = Catalog.ZeroGradient(EncoderOutputWeights.Value);
        EncoderForgetBiases.Value = Catalog.Subtract(EncoderForgetBiases.Value, Catalog.Scale(EncoderForgetBiases.Gradient, lr));
        EncoderForgetBiases.Gradient = Catalog.ZeroGradient(EncoderForgetBiases.Value);
        EncoderInputBiases.Value = Catalog.Subtract(EncoderInputBiases.Value, Catalog.Scale(EncoderInputBiases.Gradient, lr));
        EncoderInputBiases.Gradient = Catalog.ZeroGradient(EncoderInputBiases.Value);
        EncoderCellBiases.Value = Catalog.Subtract(EncoderCellBiases.Value, Catalog.Scale(EncoderCellBiases.Gradient, lr));
        EncoderCellBiases.Gradient = Catalog.ZeroGradient(EncoderCellBiases.Value);
        EncoderOutputBiases.Value = Catalog.Subtract(EncoderOutputBiases.Value, Catalog.Scale(EncoderOutputBiases.Gradient, lr));
        EncoderOutputBiases.Gradient = Catalog.ZeroGradient(EncoderOutputBiases.Value);
        
        DecoderForgetWeights.Value = Catalog.Subtract(DecoderForgetWeights.Value, Catalog.Scale(DecoderForgetWeights.Gradient, lr));
        DecoderForgetWeights.Gradient = Catalog.ZeroGradient(DecoderForgetWeights.Value);
        DecoderInputWeights.Value = Catalog.Subtract(DecoderInputWeights.Value, Catalog.Scale(DecoderInputWeights.Gradient, lr));
        DecoderInputWeights.Gradient = Catalog.ZeroGradient(DecoderInputWeights.Value);
        DecoderCellWeights.Value = Catalog.Subtract(DecoderCellWeights.Value, Catalog.Scale(DecoderCellWeights.Gradient, lr));
        DecoderCellWeights.Gradient = Catalog.ZeroGradient(DecoderCellWeights.Value);
        DecoderOutputWeights.Value = Catalog.Subtract(DecoderOutputWeights.Value, Catalog.Scale(DecoderOutputWeights.Gradient, lr));
        DecoderOutputWeights.Gradient = Catalog.ZeroGradient(DecoderOutputWeights.Value);
        DecoderForgetBiases.Value = Catalog.Subtract(DecoderForgetBiases.Value, Catalog.Scale(DecoderForgetBiases.Gradient, lr));
        DecoderForgetBiases.Gradient = Catalog.ZeroGradient(DecoderForgetBiases.Value);
        DecoderInputBiases.Value = Catalog.Subtract(DecoderInputBiases.Value, Catalog.Scale(DecoderInputBiases.Gradient, lr));
        DecoderInputBiases.Gradient = Catalog.ZeroGradient(DecoderInputBiases.Value);
        DecoderCellBiases.Value = Catalog.Subtract(DecoderCellBiases.Value, Catalog.Scale(DecoderCellBiases.Gradient, lr));
        DecoderCellBiases.Gradient = Catalog.ZeroGradient(DecoderCellBiases.Value);
        DecoderOutputBiases.Value = Catalog.Subtract(DecoderOutputBiases.Value, Catalog.Scale(DecoderOutputBiases.Gradient, lr));
        DecoderOutputBiases.Gradient = Catalog.ZeroGradient(DecoderOutputBiases.Value);
    }

    #region Read/Write
    public void Write(BinaryWriter writer)
    {
        Catalog.WriteWeights(writer, EncoderForgetWeights.Value);
        Catalog.WriteWeights(writer, EncoderInputWeights.Value);
        Catalog.WriteWeights(writer, EncoderCellWeights.Value);
        Catalog.WriteWeights(writer, EncoderOutputWeights.Value);
        
        Catalog.WriteBias(writer, EncoderForgetBiases.Value);
        Catalog.WriteBias(writer, EncoderInputBiases.Value);
        Catalog.WriteBias(writer, EncoderCellBiases.Value);
        Catalog.WriteBias(writer, EncoderOutputBiases.Value);
        
        Catalog.WriteWeights(writer, DecoderForgetWeights.Value);
        Catalog.WriteWeights(writer, DecoderInputWeights.Value);
        Catalog.WriteWeights(writer, DecoderCellWeights.Value);
        Catalog.WriteWeights(writer, DecoderOutputWeights.Value);
        
        Catalog.WriteBias(writer, DecoderForgetBiases.Value);
        Catalog.WriteBias(writer, DecoderInputBiases.Value);
        Catalog.WriteBias(writer, DecoderCellBiases.Value);
        Catalog.WriteBias(writer, DecoderOutputBiases.Value);
    }

    public void Read(BinaryReader reader)
    {
        EncoderForgetWeights.Value = Catalog.ReadWeights(reader);
        EncoderInputWeights.Value = Catalog.ReadWeights(reader);
        EncoderCellWeights.Value = Catalog.ReadWeights(reader);
        EncoderOutputWeights.Value = Catalog.ReadWeights(reader);

        EncoderForgetBiases.Value = Catalog.ReadBias(reader);
        EncoderInputBiases.Value = Catalog.ReadBias(reader);
        EncoderCellBiases.Value = Catalog.ReadBias(reader);
        EncoderOutputBiases.Value = Catalog.ReadBias(reader);

        DecoderForgetWeights.Value = Catalog.ReadWeights(reader);
        DecoderInputWeights.Value = Catalog.ReadWeights(reader);
        DecoderCellWeights.Value = Catalog.ReadWeights(reader);
        DecoderOutputWeights.Value = Catalog.ReadWeights(reader);

        DecoderForgetBiases.Value = Catalog.ReadBias(reader);
        DecoderInputBiases.Value = Catalog.ReadBias(reader);
        DecoderCellBiases.Value = Catalog.ReadBias(reader);
        DecoderOutputBiases.Value = Catalog.ReadBias(reader);  
    }
    #endregion
}

public interface ILstmCatalog<TIn, TOut, TWeights>
    where TIn : notnull where TOut : notnull
    where TWeights : notnull
{
    Tensor<TOut> ConcatInputHidden(Tensor<TIn> a, Tensor<TOut> b);
    Tensor<TOut> ConcatHidden(Tensor<TOut> a, Tensor<TOut> b);
    
    Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TOut> b);
    
    Tensor<TOut> Multiply(Tensor<TOut> a, Tensor<TOut> b);
    
    TIn ZeroGradient(TIn a);
    TOut ZeroGradient(TOut a);
    TWeights ZeroGradient(TWeights a);
    
    TWeights Subtract(TWeights a, TWeights b);
    TOut Subtract(TOut a, TOut b);
    
    TWeights Scale(TWeights a, double scale);
    TOut Scale(TOut a, double scale);
    
    TWeights ReadWeights(BinaryReader reader);
    void WriteWeights(BinaryWriter writer, TWeights weight);
    
    TOut ReadBias(BinaryReader reader);
    void WriteBias(BinaryWriter writer, TOut bias);
}