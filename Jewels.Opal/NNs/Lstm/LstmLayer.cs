using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public class LstmLayer<TIn, TOut, TWeights, TBiases> : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    public required Tensor<TWeights> EncoderForgetWeights { get; set; } 
    public required Tensor<TWeights> EncoderInputWeights { get; set; } 
    public required Tensor<TWeights> EncoderCellWeights { get; set; } 
    public required Tensor<TWeights> EncoderOutputWeights { get; set; }
    public required Tensor<TBiases> EncoderForgetBiases { get; set; }
    public required Tensor<TBiases> EncoderInputBiases { get; set; }
    public required Tensor<TBiases> EncoderCellBiases { get; set; }
    public required Tensor<TBiases> EncoderOutputBiases { get; set; }
    
    public required Tensor<TWeights> DecoderForgetWeights { get; set; } 
    public required Tensor<TWeights> DecoderInputWeights { get; set; } 
    public required Tensor<TWeights> DecoderCellWeights { get; set; } 
    public required Tensor<TWeights> DecoderOutputWeights { get; set; }
    
    public required Tensor<TBiases> DecoderForgetBiases { get; set; }
    public required Tensor<TBiases> DecoderInputBiases { get; set; }
    public required Tensor<TBiases> DecoderCellBiases { get; set; }
    public required Tensor<TBiases> DecoderOutputBiases { get; set; }
    public required ILstmCatalog<TIn, TOut, TWeights, TBiases> Catalog { get; set; }
    
    public required Tensor<TOut> DefaultState { get; set; }
    public required Tensor<TOut> DefaultHidden { get; set; }

    #region Encoder/Decoder
    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Encoder(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatInputHidden(input, prevHidden);
        
        Tensor<TOut> forgetWeighted = Catalog.Multiply(EncoderForgetWeights, concat);
        Tensor<TOut> inputWeighted = Catalog.Multiply(EncoderInputWeights, concat);
        Tensor<TOut> cellWeighted = Catalog.Multiply(EncoderCellWeights, concat);
        Tensor<TOut> outputWeighted = Catalog.Multiply(EncoderOutputWeights, concat);
        
        Tensor<TOut> forgetGate = Catalog.Sigmoid(Catalog.Add(forgetWeighted, EncoderForgetBiases));
        Tensor<TOut> inputGate = Catalog.Sigmoid(Catalog.Add(inputWeighted, EncoderInputBiases));
        Tensor<TOut> cellGate = Catalog.Tanh(Catalog.Add(cellWeighted, EncoderCellBiases));
        Tensor<TOut> outputGate = Catalog.Sigmoid(Catalog.Add(outputWeighted, EncoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.LstmState(forgetGate, state, inputGate, cellGate);
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, Catalog.Tanh(newState));
        
        return (newHidden, newState);
    }
    
    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Decoder(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden)
    {
        Tensor<TOut> concat = Catalog.ConcatHidden(input, prevHidden);
        
        Tensor<TOut> forgetWeighted = Catalog.Multiply(DecoderForgetWeights, concat);
        Tensor<TOut> inputWeighted = Catalog.Multiply(DecoderInputWeights, concat);
        Tensor<TOut> cellWeighted = Catalog.Multiply(DecoderCellWeights, concat);
        Tensor<TOut> outputWeighted = Catalog.Multiply(DecoderOutputWeights, concat);
        
        Tensor<TOut> forgetGate = Catalog.Sigmoid(Catalog.Add(forgetWeighted, DecoderForgetBiases));
        Tensor<TOut> inputGate = Catalog.Sigmoid(Catalog.Add(inputWeighted, DecoderInputBiases));
        Tensor<TOut> cellGate = Catalog.Tanh(Catalog.Add(cellWeighted, DecoderCellBiases));
        Tensor<TOut> outputGate = Catalog.Sigmoid(Catalog.Add(outputWeighted, DecoderOutputBiases));
        
        Tensor<TOut> newState = Catalog.LstmState(forgetGate, state, inputGate, cellGate);
        Tensor<TOut> newHidden = Catalog.Multiply(outputGate, Catalog.Tanh(newState));
        
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
    public Value<TOut> Forward(Value<TIn> input)
    {
        Tensor<TIn> tensorInput = new(input, input.Zeros());
        Tensor<TOut> initialHidden = DefaultHidden;
        Tensor<TOut> initialState = DefaultState;
        
        return ForwardCore(tensorInput, initialHidden, initialState).Value;
    }
    
    public Tensor<TOut> Forward(Tensor<TIn> input, Tensor<TOut> initialHidden, Tensor<TOut> initialState) => ForwardCore(input, initialHidden, initialState);
    public Tensor<TOut> Forward(Tensor<TIn> input) => ForwardCore(input, DefaultHidden, DefaultState);

    public Tensor<TOut>[] ForwardTransforming(Tensor<TIn>[] inputs)
    {
        Tensor<TOut> initialHidden = DefaultHidden;
        Tensor<TOut> initialState = DefaultState;
        var encoderOutputs = EncoderSequence(inputs, initialHidden, initialState);
        var decoderOutputs = DecoderSequence(encoderOutputs, initialHidden, initialState);
        
        return decoderOutputs;
    }
    public Tensor<TOut> ForwardSequence(Tensor<TIn>[] inputs) => ForwardTransforming(inputs).Last();
    public Value<TOut> ForwardSequence(Value<TIn>[] inputs) => ForwardSequence(inputs.Select(x => new Tensor<TIn>(x, x.Zeros())).ToArray()).Value;
    public Value<TOut>[] ForwardTransforming(Value<TIn>[] inputs) => ForwardTransforming(inputs.Select(x => new Tensor<TIn>(x, x.Zeros())).ToArray()).Select(x => x.Value).ToArray();
    #endregion

    public virtual void UpdateParameters(float lr)
    {
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderForgetWeights.Value, EncoderForgetWeights.Value, EncoderForgetWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderInputWeights.Value, EncoderInputWeights.Value, EncoderInputWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderCellWeights.Value, EncoderCellWeights.Value, EncoderCellWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderOutputWeights.Value, EncoderOutputWeights.Value, EncoderOutputWeights.Value, lr);
        
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderForgetBiases.Value, EncoderForgetBiases.Value, EncoderForgetBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderInputBiases.Value, EncoderInputBiases.Value, EncoderInputBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderCellBiases.Value, EncoderCellBiases.Value, EncoderCellBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, EncoderOutputBiases.Value, EncoderOutputBiases.Value, EncoderOutputBiases.Value, lr);
        
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderForgetWeights.Value, DecoderForgetWeights.Value, DecoderForgetWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderInputWeights.Value, DecoderInputWeights.Value, DecoderInputWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderCellWeights.Value, DecoderCellWeights.Value, DecoderCellWeights.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderOutputWeights.Value, DecoderOutputWeights.Value, DecoderOutputWeights.Value, lr);
        
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderForgetBiases.Value, DecoderForgetBiases.Value, DecoderForgetBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderInputBiases.Value, DecoderInputBiases.Value, DecoderInputBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderCellBiases.Value, DecoderCellBiases.Value, DecoderCellBiases.Value, lr);
        Operations.Compute.Call(Operations.ElementwiseFloatMulAndSubKernels, DecoderOutputBiases.Value, DecoderOutputBiases.Value, DecoderOutputBiases.Value, lr);
    }

    public virtual void ZeroGradients()
    {
        EncoderForgetWeights.Gradient.UpdateWith(EncoderForgetWeights.Gradient.Zeros());
        EncoderInputWeights.Gradient.UpdateWith(EncoderInputWeights.Gradient.Zeros());
        EncoderCellWeights.Gradient.UpdateWith(EncoderCellWeights.Gradient.Zeros());
        EncoderOutputWeights.Gradient.UpdateWith(EncoderOutputWeights.Gradient.Zeros());
        
        EncoderForgetBiases.Gradient.UpdateWith(EncoderForgetBiases.Gradient.Zeros());
        EncoderInputBiases.Gradient.UpdateWith(EncoderInputBiases.Gradient.Zeros());
        EncoderCellBiases.Gradient.UpdateWith(EncoderCellBiases.Gradient.Zeros());
        EncoderOutputBiases.Gradient.UpdateWith(EncoderOutputBiases.Gradient.Zeros());
        
        DecoderForgetWeights.Gradient.UpdateWith(DecoderForgetWeights.Gradient.Zeros());
        DecoderInputWeights.Gradient.UpdateWith(DecoderInputWeights.Gradient.Zeros());
        DecoderCellWeights.Gradient.UpdateWith(DecoderCellWeights.Gradient.Zeros());
        DecoderOutputWeights.Gradient.UpdateWith(DecoderOutputWeights.Gradient.Zeros());
        
        DecoderForgetBiases.Gradient.UpdateWith(DecoderForgetBiases.Gradient.Zeros());
        DecoderInputBiases.Gradient.UpdateWith(DecoderInputBiases.Gradient.Zeros());
        DecoderCellBiases.Gradient.UpdateWith(DecoderCellBiases.Gradient.Zeros());
        DecoderOutputBiases.Gradient.UpdateWith(DecoderOutputBiases.Gradient.Zeros());
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
        EncoderForgetWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        EncoderInputWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        EncoderCellWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        EncoderOutputWeights.Value.UpdateWith(Catalog.ReadWeights(reader));

        EncoderForgetBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        EncoderInputBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        EncoderCellBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        EncoderOutputBiases.Value.UpdateWith(Catalog.ReadBias(reader));

        DecoderForgetWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        DecoderInputWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        DecoderCellWeights.Value.UpdateWith(Catalog.ReadWeights(reader));
        DecoderOutputWeights.Value.UpdateWith(Catalog.ReadWeights(reader));

        DecoderForgetBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        DecoderInputBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        DecoderCellBiases.Value.UpdateWith(Catalog.ReadBias(reader));
        DecoderOutputBiases.Value.UpdateWith(Catalog.ReadBias(reader));  
    }
    #endregion
}

public interface ILstmCatalog<TIn, TOut, TWeights, TBiases>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    Tensor<TOut> ConcatInputHidden(Tensor<TIn> a, Tensor<TOut> b);
    Tensor<TOut> ConcatHidden(Tensor<TOut> a, Tensor<TOut> b);
    Tensor<TOut> Sigmoid(Tensor<TOut> x);
    Tensor<TOut> Tanh(Tensor<TOut> x);
    Tensor<TOut> Multiply(Tensor<TOut> a, Tensor<TOut> b);
    Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TOut> b);
    Tensor<TOut> LstmState(Tensor<TOut> forgetGate, Tensor<TOut> state, Tensor<TOut> inputGate, Tensor<TOut> cellGate);
    Tensor<TOut> Add(Tensor<TOut> a, Tensor<TBiases> b);
    
    Value<TWeights> ReadWeights(BinaryReader reader);
    void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    
    Value<TBiases> ReadBias(BinaryReader reader);
    void WriteBias(BinaryWriter writer, Value<TBiases> bias);
}