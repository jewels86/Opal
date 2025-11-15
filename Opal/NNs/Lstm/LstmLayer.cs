using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public class LstmLayer<TIn, TOut, TWeight> : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeight : notnull
{
    public required Tensor<TWeight>[] EncoderForgetWeights { get; set; } 
    public required Tensor<TWeight>[] EncoderInputWeights { get; set; } 
    public required Tensor<TWeight>[] EncoderCellWeights { get; set; } 
    public required Tensor<TWeight>[] EncoderOutputWeights { get; set; }
    public required Tensor<TOut> EncoderForgetBiases { get; set; }
    public required Tensor<TOut> EncoderInputBiases { get; set; }
    public required Tensor<TOut> EncoderCellBiases { get; set; }
    public required Tensor<TOut> EncoderOutputBiases { get; set; }
    
    public required Tensor<TWeight>[] DecoderForgetWeights { get; set; } 
    public required Tensor<TWeight>[] DecoderInputWeights { get; set; } 
    public required Tensor<TWeight>[] DecoderCellWeights { get; set; } 
    public required Tensor<TWeight>[] DecoderOutputWeights { get; set; }
    
    public required Tensor<TOut> DecoderForgetBiases { get; set; }
    public required Tensor<TOut> DecoderInputBiases { get; set; }
    public required Tensor<TOut> DecoderCellBiases { get; set; }
    public required Tensor<TOut> DecoderOutputBiases { get; set; }
    
    public required ActivationFunction<TOut> SigmoidActivation { get; set; }
    public required ActivationFunction<TOut> TanhActivation { get; set; }
    public required ILstmCatalog<TIn, TOut, TWeight> Catalog { get; set; }

    #region Encoder/Decoder
    public (Tensor<TOut> hidden, Tensor<TOut> state) EncoderForward(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatInputHidden(input, prevHidden);
        Tensor<TOut> forgetGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, EncoderForgetWeights), EncoderForgetBiases));
        Tensor<TOut> inputGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, EncoderInputWeights), EncoderInputBiases));
        Tensor<TOut> cellGate = TanhActivation.Function(Catalog.Add(Catalog.Multiply(concat, EncoderCellWeights), EncoderCellBiases));
        Tensor<TOut> outputGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, EncoderOutputWeights), EncoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.Add(Catalog.Multiply(forgetGate, state), Catalog.Multiply(inputGate, cellGate));
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, TanhActivation.Function(newState));
        
        return (newHidden, newState);
    }
    
    public (Tensor<TOut> hidden, Tensor<TOut> state) DecoderForward(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatHidden(input, prevHidden);
        Tensor<TOut> forgetGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, DecoderForgetWeights), DecoderForgetBiases));
        Tensor<TOut> inputGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, DecoderInputWeights), DecoderInputBiases));
        Tensor<TOut> cellGate = TanhActivation.Function(Catalog.Add(Catalog.Multiply(concat, DecoderCellWeights), DecoderCellBiases));
        Tensor<TOut> outputGate = SigmoidActivation.Function(Catalog.Add(Catalog.Multiply(concat, DecoderOutputWeights), DecoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.Add(Catalog.Multiply(forgetGate, state), Catalog.Multiply(inputGate, cellGate));
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, TanhActivation.Function(newState));
        
        return (newHidden, newState);
    }

    public Tensor<TOut>[] EncoderForwardSequence(Tensor<TIn>[] inputs, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        List<Tensor<TOut>> hiddenStates = new();
        Tensor<TOut> currentHidden = initialHidden;
        Tensor<TOut> currentState = initialState;
        
        foreach (var input in inputs)
        {
            var (hidden, state) = EncoderForward(input, currentState, currentHidden);
            hiddenStates.Add(hidden);
            currentHidden = hidden;
            currentState = state;
        }
        
        return hiddenStates.ToArray();
    }
    
    public Tensor<TOut>[] DecoderForwardSequence(Tensor<TOut>[] inputs, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        List<Tensor<TOut>> hiddenStates = new();
        Tensor<TOut> currentHidden = initialHidden;
        Tensor<TOut> currentState = initialState;
        
        foreach (var input in inputs)
        {
            var (hidden, state) = DecoderForward(input, currentState, currentHidden);
            hiddenStates.Add(hidden);
            currentHidden = hidden;
            currentState = state;
        }
        
        return hiddenStates.ToArray();
    }
    #endregion

    public virtual Tensor<TOut> ForwardCore(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState)
    {
        var encoderOutput = EncoderForward(input, initialState, initialHidden);
        var decoderOutput = DecoderForward(encoderOutput.hidden, encoderOutput.state, encoderOutput.hidden);
        return decoderOutput.hidden;
    }
    
    #region Overloads
    public TOut Forward(TIn input)
    {
        Tensor<TIn> tensorInput = new(input, null, _ => { }, Catalog.ZeroGradient(input));
        Tensor<TOut> initialHidden = Catalog.DefaultHidden();
        Tensor<TOut> initialState = Catalog.DefaultState();
        
        return ForwardCore(tensorInput, initialHidden, initialState).Value;
    }
    
    public Tensor<TOut> Forward(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState) => ForwardCore(input, initialHidden, initialState);
    public Tensor<TOut> Forward(Tensor<TIn> input) => ForwardCore(input, Catalog.DefaultHidden(), Catalog.DefaultState());

    public TOut ForwardSequence(TIn[] inputs)
    {
        Tensor<TOut> initialHidden = Catalog.DefaultHidden();
        Tensor<TOut> initialState = Catalog.DefaultState();
        
        var tensorInputs = inputs.Select(i => new Tensor<TIn>(i, null, _ => { }, Catalog.ZeroGradient(i))).ToArray();
        var encoderOutputs = EncoderForwardSequence(tensorInputs, initialHidden, initialState);
        var decoderOutputs = DecoderForwardSequence(encoderOutputs, initialHidden, initialState);
        
        return decoderOutputs[^1].Value;
    }
    #endregion

    #region Read/Write
    public void Write(BinaryWriter writer)
    {
        writer.Write(EncoderForgetWeights.Length);
        foreach (var weight in EncoderForgetWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(EncoderInputWeights.Length);
        foreach (var weight in EncoderInputWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(EncoderCellWeights.Length);
        foreach (var weight in EncoderCellWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(EncoderOutputWeights.Length);
        foreach (var weight in EncoderOutputWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        Catalog.WriteBias(writer, EncoderForgetBiases.Value);
        Catalog.WriteBias(writer, EncoderInputBiases.Value);
        Catalog.WriteBias(writer, EncoderCellBiases.Value);
        Catalog.WriteBias(writer, EncoderOutputBiases.Value);
        
        writer.Write(DecoderForgetWeights.Length);
        foreach (var weight in DecoderForgetWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(DecoderInputWeights.Length);
        foreach (var weight in DecoderInputWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(DecoderCellWeights.Length);
        foreach (var weight in DecoderCellWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        writer.Write(DecoderOutputWeights.Length);
        foreach (var weight in DecoderOutputWeights)
            Catalog.WriteWeight(writer, weight.Value);
        
        Catalog.WriteBias(writer, DecoderForgetBiases.Value);
        Catalog.WriteBias(writer, DecoderInputBiases.Value);
        Catalog.WriteBias(writer, DecoderCellBiases.Value);
        Catalog.WriteBias(writer, DecoderOutputBiases.Value);
    }

    public void Read(BinaryReader reader)
    {
        int encoderForgetWeightsLength = reader.ReadInt32();
        EncoderForgetWeights = new Tensor<TWeight>[encoderForgetWeightsLength];
        for (int i = 0; i < encoderForgetWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            EncoderForgetWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int encoderInputWeightsLength = reader.ReadInt32();
        EncoderInputWeights = new Tensor<TWeight>[encoderInputWeightsLength];
        for (int i = 0; i < encoderInputWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            EncoderInputWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int encoderCellWeightsLength = reader.ReadInt32();
        EncoderCellWeights = new Tensor<TWeight>[encoderCellWeightsLength];
        for (int i = 0; i < encoderCellWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            EncoderCellWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int encoderOutputWeightsLength = reader.ReadInt32();
        EncoderOutputWeights = new Tensor<TWeight>[encoderOutputWeightsLength];
        for (int i = 0; i < encoderOutputWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            EncoderOutputWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        EncoderForgetBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        EncoderInputBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        EncoderCellBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        EncoderOutputBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        
        int decoderForgetWeightsLength = reader.ReadInt32();
        DecoderForgetWeights = new Tensor<TWeight>[decoderForgetWeightsLength];
        for (int i = 0; i < decoderForgetWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            DecoderForgetWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int decoderInputWeightsLength = reader.ReadInt32();
        DecoderInputWeights = new Tensor<TWeight>[decoderInputWeightsLength];
        for (int i = 0; i < decoderInputWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            DecoderInputWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int decoderCellWeightsLength = reader.ReadInt32();
        DecoderCellWeights = new Tensor<TWeight>[decoderCellWeightsLength];
        for (int i = 0; i < decoderCellWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            DecoderCellWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        int decoderOutputWeightsLength = reader.ReadInt32();
        DecoderOutputWeights = new Tensor<TWeight>[decoderOutputWeightsLength];
        for (int i = 0; i < decoderOutputWeightsLength; i++)
        {
            var weight = Catalog.ReadWeight(reader);
            DecoderOutputWeights[i] = new Tensor<TWeight>(weight, null, _ => { }, Catalog.ZeroGradient(weight));
        }
        
        DecoderForgetBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        DecoderInputBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        DecoderCellBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
        DecoderOutputBiases = new Tensor<TOut>(Catalog.ReadBias(reader), null, _ => { }, Catalog.ZeroGradient(Catalog.ReadBias(reader)));
    }
    #endregion
}

public interface ILstmCatalog<TIn, TOut, TWeight>
    where TIn : notnull where TOut : notnull
    where TWeight : notnull
{
    Tensor<TOut> ConcatInputHidden(Tensor<TIn> input, Tensor<TOut> prevHidden);
    Tensor<TOut> ConcatHidden(Tensor<TOut> input, Tensor<TOut> prevHidden);
    
    Tensor<TOut> Add(Tensor<TOut> a, Tensor<TOut> b);
    Tensor<TOut> Multiply(Tensor<TOut> concat, Tensor<TWeight>[] weights);
    
    Tensor<TOut> Multiply(Tensor<TOut> a, Tensor<TOut> b);
    
    Tensor<TOut> DefaultHidden();
    Tensor<TOut> DefaultState();
    
    TIn ZeroGradient(TIn a);
    TOut ZeroGradient(TOut a);
    TWeight ZeroGradient(TWeight a);
    
    TWeight ReadWeight(BinaryReader reader);
    void WriteWeight(BinaryWriter writer, TWeight weight);
    
    TOut ReadBias(BinaryReader reader);
    void WriteBias(BinaryWriter writer, TOut bias);
}