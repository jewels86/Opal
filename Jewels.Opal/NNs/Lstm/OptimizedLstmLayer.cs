namespace Jewels.Opal.NNs;

public class OptimizedLstmLayer<TIn, TOut, TWeights, TBiases> : LstmLayer<TIn, TOut, TWeights, TBiases>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    public required IOptimizedLstmCatalog<TIn, TOut, TWeights, TBiases> OptimizedCatalog { get; init; } 

    #region Encoder/Decoder
    public override (Tensor<TOut> hidden, Tensor<TOut> state) Encoder(Tensor<TIn> input, Tensor<TOut> state, Tensor<TOut> prevHidden) => 
        OptimizedCatalog.InLstmUpdate(input, prevHidden, state, EncoderParameters);

    public override (Tensor<TOut> hidden, Tensor<TOut> state) Decoder(Tensor<TOut> input, Tensor<TOut> state, Tensor<TOut> prevHidden) => 
        OptimizedCatalog.OutLstmUpdate(input, prevHidden, state, DecoderParameters);
    #endregion
    
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

}

public interface IOptimizedLstmCatalog<TIn, TOut, TWeights, TBiases>
    where TIn : notnull where TOut : notnull where TWeights : notnull where TBiases : notnull
{
    (Tensor<TOut>, Tensor<TOut>) InLstmUpdate(Tensor<TIn> input, Tensor<TOut> hidden, Tensor<TOut> prevState, LstmUpdateParameters<TWeights, TBiases> parameters); 
    (Tensor<TOut>, Tensor<TOut>) OutLstmUpdate(Tensor<TOut> input, Tensor<TOut> hidden, Tensor<TOut> state, LstmUpdateParameters<TWeights, TBiases> parameters);
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