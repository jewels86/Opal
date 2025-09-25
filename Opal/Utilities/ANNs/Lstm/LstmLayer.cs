using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Lstm;

using static MathFunctions;
using static Logging;
using static BinaryWriting;

public class LstmLayer<TWeights, TBiases, TTensor> : ILayer<TTensor, TTensor> 
    where TWeights : notnull where TBiases : notnull
    where TTensor : notnull

{
    public string Name { get; }
    
    public int[] InputShape { get; }
    public int[] HiddenShape { get; }
    public int[] OutputShape { get; }
    
    public TWeights EncoderForgetGateWeights { get; set; }
    public TWeights EncoderInputGateWeights { get; set; }
    public TWeights EncoderOutputGateWeights { get; set; }
    public TWeights EncoderCellGateWeights { get; set; }
    public TBiases EncoderForgetGateBiases { get; set; }
    public TBiases EncoderInputGateBiases { get; set; }
    public TBiases EncoderOutputGateBiases { get; set; }
    public TBiases EncoderCellGateBiases { get; set; }
    
    public TWeights DecoderForgetGateWeights { get; set; }
    public TWeights DecoderInputGateWeights { get; set; }
    public TWeights DecoderOutputGateWeights { get; set; }
    public TWeights DecoderCellGateWeights { get; set; }
    public TBiases DecoderForgetGateBiases { get; set; }
    public TBiases DecoderInputGateBiases { get; set; }
    public TBiases DecoderOutputGateBiases { get; set; }
    public TBiases DecoderCellGateBiases { get; set; }
    
    private List<TTensor> encoderInputCache = [];
    private List<TTensor> encoderForgetCache = [];
    private List<TTensor> encoderInputGateCache = [];
    private List<TTensor> encoderOutputGateCache = [];
    private List<TTensor> encoderCellGateCache = [];
    private List<TTensor> encoderNewCellCache = [];
    private List<TTensor> encoderNewHiddenCache = [];
    
    private List<TTensor> decoderInputCache = [];
    private List<TTensor> decoderForgetCache = [];
    private List<TTensor> decoderInputGateCache = [];
    private List<TTensor> decoderOutputGateCache = [];
    private List<TTensor> decoderCellGateCache = [];
    private List<TTensor> decoderNewCellCache = [];
    private List<TTensor> decoderNewHiddenCache = [];
    
    public ActivationFunction<TTensor> SigmoidActivation { get; set; }
    public ActivationFunction<TTensor> TanhActivation { get; set; }

    private readonly ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations;
    private readonly IOptimizer<TWeights, TBiases> optimizer;

    public LstmLayer(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation,
        string? name = null)
    {
        Name = name ?? "LstmLayer";
        
        InputShape = inputShape;
        HiddenShape = hiddenShape;
        OutputShape = outputShape;
        
        this.tensorOperations = tensorOperations;
        this.optimizer = optimizer;
        
        EncoderForgetGateWeights = tensorOperations.DefaultWeights(hiddenShape, inputShape);
        EncoderInputGateWeights = tensorOperations.DefaultWeights(hiddenShape, inputShape);
        EncoderOutputGateWeights = tensorOperations.DefaultWeights(hiddenShape, inputShape);
        EncoderCellGateWeights = tensorOperations.DefaultWeights(hiddenShape, inputShape);
        EncoderForgetGateBiases = tensorOperations.DefaultBiases(hiddenShape);
        EncoderInputGateBiases = tensorOperations.DefaultBiases(hiddenShape);
        EncoderOutputGateBiases = tensorOperations.DefaultBiases(hiddenShape);
        EncoderCellGateBiases = tensorOperations.DefaultBiases(hiddenShape);
        
        DecoderForgetGateWeights = tensorOperations.DefaultWeights(outputShape, hiddenShape);
        DecoderInputGateWeights = tensorOperations.DefaultWeights(outputShape, hiddenShape);
        DecoderOutputGateWeights = tensorOperations.DefaultWeights(outputShape, hiddenShape);
        DecoderCellGateWeights = tensorOperations.DefaultWeights(outputShape, hiddenShape);
        DecoderForgetGateBiases = tensorOperations.DefaultBiases(outputShape);
        DecoderInputGateBiases = tensorOperations.DefaultBiases(outputShape);
        DecoderOutputGateBiases = tensorOperations.DefaultBiases(outputShape);
        DecoderCellGateBiases = tensorOperations.DefaultBiases(outputShape);
        
        SigmoidActivation = sigmoidActivation;
        TanhActivation = tanhActivation;
    }

    #region Encoder
    public delegate void EncoderCacheHandler(TTensor input, TTensor forget, TTensor inputGate, TTensor outputGate, TTensor cellGate, TTensor newCell, TTensor newHidden);

    public (TTensor hidden, TTensor cell) Encoder(TTensor input, TTensor hidden, TTensor cell, EncoderCacheHandler? cacheHandler = null)
    {
        TTensor concat = tensorOperations.Concat(input, hidden);
        TTensor forget = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(EncoderForgetGateWeights, concat), EncoderForgetGateBiases));
        TTensor inputGate = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(EncoderInputGateWeights, concat), EncoderInputGateBiases));
        TTensor outputGate = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(EncoderOutputGateWeights, concat), EncoderOutputGateBiases));
        TTensor cellGate = TanhActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(EncoderCellGateWeights, concat), EncoderCellGateBiases));
        TTensor newCell = tensorOperations.Add(tensorOperations.Multiply(forget, cell), tensorOperations.Multiply(inputGate, cellGate));
        TTensor newHidden = tensorOperations.Multiply(outputGate, TanhActivation.Function(newCell));
        if (cacheHandler is not null)
            cacheHandler(input, forget, inputGate, outputGate, cellGate, newCell, newHidden);
        return (newHidden, newCell);
    }

    public TTensor[] Encoder(TTensor[] inputs, bool cache = true)
    {
        List<TTensor> hiddenStates = [tensorOperations.DefaultState(HiddenShape)];
        List<TTensor> cellStates = [tensorOperations.DefaultCell(OutputShape)];
        
        EncoderCacheHandler? cacheFunc = !cache ? null : (input, forget, inputGate, outputGate, cellGate, newCell, newHidden) =>
        {
            encoderInputCache.Add(input);
            encoderForgetCache.Add(forget);
            encoderInputGateCache.Add(inputGate);
            encoderOutputGateCache.Add(outputGate);
            encoderCellGateCache.Add(cellGate);
            encoderNewCellCache.Add(newCell);
            encoderNewHiddenCache.Add(newHidden);
        };
        
        int timeSteps = inputs.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            TTensor input = inputs[t];
            var (hidden, cell) = Encoder(input, hiddenStates.Last(), cellStates.Last(), cacheFunc);
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }
        
        return hiddenStates.Skip(1).ToArray();
    }
    #endregion
    #region Decoder

    public delegate void DecoderCacheHandler(TTensor input, TTensor forget, TTensor inputGate, TTensor outputGate, TTensor cellGate, TTensor newCell, TTensor newHidden);
    public (TTensor hidden, TTensor cell) Decoder(TTensor input, TTensor hidden, TTensor cell, DecoderCacheHandler? cacheHandler = null)
    {
        TTensor concat = tensorOperations.Concat(input, hidden);
        TTensor forget = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(DecoderForgetGateWeights, concat), DecoderForgetGateBiases));
        TTensor inputGate = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(DecoderInputGateWeights, concat), DecoderInputGateBiases));
        TTensor outputGate = SigmoidActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(DecoderOutputGateWeights, concat), DecoderOutputGateBiases));
        TTensor cellGate = TanhActivation.Function(tensorOperations.Add(
            tensorOperations.Multiply(DecoderCellGateWeights, concat), DecoderCellGateBiases));
        TTensor newCell = tensorOperations.Add(tensorOperations.Multiply(forget, cell), tensorOperations.Multiply(inputGate, cellGate));
        TTensor newHidden = tensorOperations.Multiply(outputGate, TanhActivation.Function(newCell));
        if (cacheHandler is not null)
            cacheHandler(input, forget, inputGate, outputGate, cellGate, newCell, newHidden);
        return (newHidden, newCell);
    }
    
    public TTensor[] Decoder(TTensor[] inputs, TTensor initialHidden, TTensor initialCell, bool cache = true)
    {
        List<TTensor> hiddenStates = [initialHidden];
        List<TTensor> cellStates = [initialCell];
        
        DecoderCacheHandler? cacheFunc = !cache ? null : (input, forget, inputGate, outputGate, cellGate, newCell, newHidden) =>
        {
            decoderInputCache.Add(input);
            decoderForgetCache.Add(forget);
            decoderInputGateCache.Add(inputGate);
            decoderOutputGateCache.Add(outputGate);
            decoderCellGateCache.Add(cellGate);
            decoderNewCellCache.Add(newCell);
            decoderNewHiddenCache.Add(newHidden);
        };
        
        int timeSteps = inputs.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            TTensor input = inputs[t];
            var (hidden, cell) = Decoder(input, hiddenStates.Last(), cellStates.Last(), cacheFunc);
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }
        
        return hiddenStates.Skip(1).ToArray();
    }
    #endregion
    
    public TTensor Forward(TTensor input)
    {
        var encoderOutputs = Encoder([input]);
        var decoderOutputs = Decoder(encoderOutputs, tensorOperations.DefaultState(HiddenShape), tensorOperations.DefaultCell(OutputShape));
        return decoderOutputs.Last();
    }
    
    #region Backward
    
    #endregion
}

public interface ILstmTensorOperations<TWeights, TBiases, TTensor>
{
    public TWeights DefaultWeights(int[] outputShape, int[] inputShape);
    public TBiases DefaultBiases(int[] outputShape);
    public TTensor DefaultState(int[] shape);
    public TTensor DefaultCell(int[] shape);
    
    public TTensor Concat(TTensor a, TTensor b);
    public TTensor Multiply(TWeights weights, TTensor input);
    public TTensor Add(TTensor a, TBiases b);
    public TTensor Add(TTensor a, TTensor b);
    public TTensor Multiply(TTensor a, TTensor b);
}