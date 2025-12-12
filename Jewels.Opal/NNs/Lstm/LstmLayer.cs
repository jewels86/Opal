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
    public required ILstmCatalog<TIn, TOut, TWeights, TBiases> Catalog { get; init; }
    
    public required Tensor<TOut> EncoderHidden { get; set; }
    public required Tensor<TOut> EncoderState { get; set; }
    public required Tensor<TOut> DecoderHidden { get; set; }
    public required Tensor<TOut> DecoderState { get; set; }

    #region Encoder/Decoder
    public virtual Tensor<TOut> Encoder(Tensor<TIn> input)
    {
        Tensor<TOut> concat = Catalog.ConcatInputHidden(input, EncoderHidden);
        
        Tensor<TOut> forgetWeighted = Catalog.Multiply(EncoderForgetWeights, concat);
        Tensor<TOut> inputWeighted = Catalog.Multiply(EncoderInputWeights, concat);
        Tensor<TOut> cellWeighted = Catalog.Multiply(EncoderCellWeights, concat);
        Tensor<TOut> outputWeighted = Catalog.Multiply(EncoderOutputWeights, concat);
        
        Tensor<TOut> forgetGate = Catalog.LstmSigmoidGate(forgetWeighted, EncoderForgetBiases);
        Tensor<TOut> inputGate = Catalog.LstmSigmoidGate(inputWeighted, EncoderInputBiases);
        Tensor<TOut> cellGate = Catalog.LstmTanhGate(cellWeighted, EncoderCellBiases);
        Tensor<TOut> outputGate = Catalog.LstmSigmoidGate(outputWeighted, EncoderOutputBiases);
        
        EncoderState = Catalog.LstmState(forgetGate, EncoderState, inputGate, cellGate);
        EncoderHidden = Catalog.LstmHidden(outputGate, EncoderState);
        
        return EncoderHidden;
    }
    
    public virtual Tensor<TOut> Decoder(Tensor<TOut> input)
    {
        Tensor<TOut> concat = Catalog.ConcatHidden(input, DecoderHidden);
        
        Tensor<TOut> forgetWeighted = Catalog.Multiply(DecoderForgetWeights, concat);
        Tensor<TOut> inputWeighted = Catalog.Multiply(DecoderInputWeights, concat);
        Tensor<TOut> cellWeighted = Catalog.Multiply(DecoderCellWeights, concat);
        Tensor<TOut> outputWeighted = Catalog.Multiply(DecoderOutputWeights, concat);
        
        Tensor<TOut> forgetGate = Catalog.LstmSigmoidGate(forgetWeighted, DecoderForgetBiases);
        Tensor<TOut> inputGate = Catalog.LstmSigmoidGate(inputWeighted, DecoderInputBiases);
        Tensor<TOut> cellGate = Catalog.LstmTanhGate(cellWeighted, DecoderCellBiases);
        Tensor<TOut> outputGate = Catalog.LstmSigmoidGate(outputWeighted, DecoderOutputBiases);
        
        DecoderState = Catalog.LstmState(forgetGate, DecoderState, inputGate, cellGate);
        DecoderHidden = Catalog.LstmHidden(outputGate, DecoderState);

        return DecoderHidden;
    }
    #endregion

    public virtual Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var encoderOutput = Encoder(input);
        var decoderOutput = Decoder(encoderOutput);
        return decoderOutput;
    }
    
    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input)).Value;

    public virtual void UpdateParameters(float lr)
    {
        Operations.Sgd(Weights, lr);
        Operations.Sgd(Biases, lr);
        ZeroGradients();
    }

    public virtual void ZeroGradients()
    {
        Operations.ZeroGradients(Weights);
        Operations.ZeroGradients(Biases);
    }

    public virtual void ResetState()
    {
        EncoderState = new(EncoderState.Value.Zeros());
        DecoderState = new(DecoderState.Value.Zeros());
        EncoderHidden = new(EncoderHidden.Value.Zeros());
        DecoderHidden = new(DecoderHidden.Value.Zeros());
    }

    public virtual ITensor[] Weights =>
    [
        EncoderForgetWeights, EncoderInputWeights, EncoderCellWeights, EncoderOutputWeights,
        DecoderForgetWeights, DecoderInputWeights, DecoderCellWeights, DecoderOutputWeights
    ];
    public virtual ITensor[] Biases =>
    [
        EncoderForgetBiases, EncoderInputBiases, EncoderCellBiases, EncoderOutputBiases,
        DecoderForgetBiases, DecoderInputBiases, DecoderCellBiases, DecoderOutputBiases
    ];
    public virtual ITensor[] Parameters =>
    [
        EncoderForgetWeights, EncoderInputWeights, EncoderCellWeights, EncoderOutputWeights,
        EncoderForgetBiases, EncoderInputBiases, EncoderCellBiases, EncoderOutputBiases,
        DecoderForgetWeights, DecoderInputWeights, DecoderCellWeights, DecoderOutputWeights,
        DecoderForgetBiases, DecoderInputBiases, DecoderCellBiases, DecoderOutputBiases
    ];
    public virtual ITensor[] States => [EncoderState, DecoderState, EncoderHidden, DecoderHidden];
    public virtual ITensor[] AllParameters => Parameters.Concat(States).ToArray();

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
    Tensor<TOut> Multiply(Tensor<TWeights> a, Tensor<TOut> b);
    Tensor<TOut> LstmState(Tensor<TOut> forgetGate, Tensor<TOut> state, Tensor<TOut> inputGate, Tensor<TOut> cellGate); // (forgetGate * state) + (inputGate * cellGate)
    Tensor<TOut> LstmHidden(Tensor<TOut> outputGate, Tensor<TOut> newState);  // outputGate * Tanh(newState)
    Tensor<TOut> LstmSigmoidGate(Tensor<TOut> weighted, Tensor<TBiases> bias); // Sigmoid(weighted + bias)
    Tensor<TOut> LstmTanhGate(Tensor<TOut> weighted, Tensor<TBiases> bias); // Tanh(weighted + bias)
    
    Value<TWeights> ReadWeights(BinaryReader reader);
    void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    
    Value<TBiases> ReadBias(BinaryReader reader);
    void WriteBias(BinaryWriter writer, Value<TBiases> bias);
}