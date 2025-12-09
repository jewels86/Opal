using System.Data;
using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public class OptimizedLstmLayer<TIn, TOut, TWeights, TBiases> : ILayer<TIn, TOut>
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
    public required IOptimizedLstmCatalog<TIn, TOut, TWeights, TBiases> Catalog { get; set; }
    
    public required Tensor<TOut> DefaultState { get; set; }
    public required Tensor<TOut> DefaultHidden { get; set; }

    #region Encoder/Decoder
    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Encoder(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden) => 
        Catalog.InLstmUpdate(input, prevHidden, state, EncoderParameters);

    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) Decoder(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden) => 
        Catalog.OutLstmUpdate(input, prevHidden, state, DecoderParameters);

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

    public virtual (Tensor<TOut> hidden, Tensor<TOut> state) ForwardWithState(Tensor<TIn> input, Tensor<TOut> hidden, Tensor<TOut> state)
    {
        var encoderOutput = Encoder(input, state, hidden);
        var decoderOutput = Decoder(encoderOutput.hidden, encoderOutput.state, encoderOutput.hidden);
        return decoderOutput;
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
        Operations.Sgd(Weights, lr);
        Operations.Sgd(Biases, lr);
        ZeroGradients();
    }

    public virtual void ZeroGradients()
    {
        Operations.ZeroGradients(Weights);
        Operations.ZeroGradients(Biases);
    }

    public virtual Tensor<TWeights>[] Weights =>
    [
        EncoderForgetWeights, EncoderInputWeights, EncoderCellWeights, EncoderOutputWeights,
        DecoderForgetWeights, DecoderInputWeights, DecoderCellWeights, DecoderOutputWeights
    ];
    public virtual Tensor<TBiases>[] Biases =>
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

    public LstmUpdateParameters<TWeights, TBiases> EncoderParameters => new()
    {
        ForgetWeights = EncoderForgetWeights, InputWeights = EncoderInputWeights, CellWeights = EncoderCellWeights, OutputWeights = EncoderOutputWeights,
        ForgetBiases = EncoderForgetBiases, InputBiases = EncoderInputBiases, CellBiases = EncoderCellBiases, OutputBiases = EncoderOutputBiases
    };
    public LstmUpdateParameters<TWeights, TBiases> DecoderParameters => new()
    {
        ForgetWeights = DecoderForgetWeights, InputWeights = DecoderInputWeights, CellWeights = DecoderCellWeights, OutputWeights = DecoderOutputWeights,
        ForgetBiases = DecoderForgetBiases, InputBiases = DecoderInputBiases, CellBiases = DecoderCellBiases, OutputBiases = DecoderOutputBiases
    };

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

public interface IOptimizedLstmCatalog<TIn, TOut, TWeights, TBiases>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    (Tensor<TOut>, Tensor<TOut>) InLstmUpdate(Tensor<TIn> input, Tensor<TOut> hidden, Tensor<TOut> state, LstmUpdateParameters<TWeights, TBiases> parameters); 
    (Tensor<TOut>, Tensor<TOut>) OutLstmUpdate(Tensor<TOut> input, Tensor<TOut> hidden, Tensor<TOut> state, LstmUpdateParameters<TWeights, TBiases> parameters);
    
    Value<TWeights> ReadWeights(BinaryReader reader);
    void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    
    Value<TBiases> ReadBias(BinaryReader reader);
    void WriteBias(BinaryWriter writer, Value<TBiases> bias);
}

public struct LstmUpdateParameters<TWeights, TBiases> where TWeights : notnull where TBiases : notnull
{
    public Tensor<TWeights> ForgetWeights { get; set; }
    public Tensor<TWeights> InputWeights { get; set; }
    public Tensor<TWeights> CellWeights { get; set; }
    public Tensor<TWeights> OutputWeights { get; set; }
    
    public Tensor<TBiases> ForgetBiases { get; set; }
    public Tensor<TBiases> InputBiases { get; set; }
    public Tensor<TBiases> CellBiases { get; set; }
    public Tensor<TBiases> OutputBiases { get; set; }
}