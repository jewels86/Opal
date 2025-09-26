using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Lstm;

public class LstmLayer<TWeights, TBiases, TTensor> : ILayer<TTensor[], TTensor[]>
    where TWeights : notnull where TBiases : notnull
    where TTensor : notnull

{
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
    
    protected List<TTensor> encoderInputCache = [];
    protected List<TTensor> encoderForgetCache = [];
    protected List<TTensor> encoderInputGateCache = [];
    protected List<TTensor> encoderOutputGateCache = [];
    protected List<TTensor> encoderCellGateCache = [];
    protected List<TTensor> encoderNewCellCache = [];
    protected List<TTensor> encoderNewHiddenCache = [];
    
    protected List<TTensor> decoderInputCache = [];
    protected List<TTensor> decoderForgetCache = [];
    protected List<TTensor> decoderInputGateCache = [];
    protected List<TTensor> decoderOutputGateCache = [];
    protected List<TTensor> decoderCellGateCache = [];
    protected List<TTensor> decoderNewCellCache = [];
    protected List<TTensor> decoderNewHiddenCache = [];

    protected readonly ActivationFunction<TTensor> SigmoidActivation;
    protected readonly ActivationFunction<TTensor> TanhActivation;

    protected readonly ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations;
    protected readonly IOptimizer<TWeights, TBiases> optimizer;

    public LstmLayer(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation)
    {
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

    public virtual TTensor[] Forward(TTensor[] inputs, TTensor initialHidden, TTensor initialCell, bool cache = true)
    {
        var encoderOutputs = Encoder(inputs, cache);
        var decoderOutputs = Decoder(encoderOutputs, initialHidden, initialCell, cache);
        return decoderOutputs;
    }
    
    public TTensor[] Forward(TTensor[] inputs, bool cache) => Forward(inputs, tensorOperations.DefaultState(HiddenShape), tensorOperations.DefaultCell(OutputShape), cache);
    public TTensor[] Forward(TTensor[] inputs) => Forward(inputs, true);
    
    #region Backward
    public TTensor[] DecoderBackward(TTensor[] gradOutputs, double learningRate)
    {
        int timeSteps = gradOutputs.GetLength(0);
        var dWForget = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWInput = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWOutput = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWCell = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dbForget = tensorOperations.DefaultBiases(OutputShape);
        var dbInput = tensorOperations.DefaultBiases(OutputShape);
        var dbOutput = tensorOperations.DefaultBiases(OutputShape);
        var dbCell = tensorOperations.DefaultBiases(OutputShape);
        
        TTensor dHiddenNext = tensorOperations.DefaultState(OutputShape);
        TTensor dCellNext = tensorOperations.DefaultCell(OutputShape);
        List<TTensor> dHiddenStates = [];

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            TTensor cell = decoderNewCellCache[t];
            TTensor prevCell = t == 0 ? tensorOperations.DefaultCell(OutputShape) : decoderNewCellCache[t - 1];
            TTensor input = decoderInputCache[t];
            TTensor forget = decoderForgetCache[t];
            TTensor inputGate = decoderInputGateCache[t];
            TTensor outputGate = decoderOutputGateCache[t];
            TTensor cellGate = decoderCellGateCache[t];
            
            TTensor dHidden = tensorOperations.Add(gradOutputs[t], dHiddenNext);
            TTensor tanhCell = TanhActivation.Function(cell);
            TTensor dOutputGate = tensorOperations.Multiply(dHidden, tanhCell);
            TTensor dOutputGatePre = tensorOperations.Multiply(dOutputGate, SigmoidActivation.Derivative(outputGate));
            
            TTensor dCell = tensorOperations.Add(tensorOperations.Multiply(dHidden, outputGate, TanhActivation.Derivative(cell)), dCellNext);

            var dInputGate = tensorOperations.Multiply(dCell, cellGate);
            var dInputGatePre = tensorOperations.Multiply(dInputGate, SigmoidActivation.Derivative(inputGate));

            var dCellCandidate = tensorOperations.Multiply(dCell, inputGate);
            var dCellCandidatePre = tensorOperations.Multiply(dCellCandidate, TanhActivation.Derivative(cellGate));

            var dForgetGate = tensorOperations.Multiply(dCell, prevCell);
            var dForgetGatePre = tensorOperations.Multiply(dForgetGate, SigmoidActivation.Derivative(forget));
            
            TTensor concat = tensorOperations.Concat(input, t == 0 ? tensorOperations.DefaultState(HiddenShape) : decoderNewHiddenCache[t - 1]);
            
            tensorOperations.UpdateAccumulatedWeights(dWForget, dForgetGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWInput, dInputGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWOutput, dOutputGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWCell, dCellCandidatePre, concat);
            tensorOperations.UpdateAccumulatedBiases(dbForget, dForgetGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbInput, dInputGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbOutput, dOutputGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbCell, dCellCandidatePre);

            dCellNext = tensorOperations.Multiply(dCell, forget);
            dHiddenNext = tensorOperations.Add(
                tensorOperations.Multiply(DecoderForgetGateWeights, dForgetGatePre),
                tensorOperations.Multiply(DecoderInputGateWeights, dInputGatePre),
                tensorOperations.Multiply(DecoderOutputGateWeights, dOutputGatePre),
                tensorOperations.Multiply(DecoderCellGateWeights, dCellCandidatePre)
            );
            
            dHiddenStates.Add(dHiddenNext);
        }
        
        DecoderForgetGateWeights = optimizer.UpdateWeights(DecoderForgetGateWeights, dWForget, learningRate);
        DecoderInputGateWeights = optimizer.UpdateWeights(DecoderInputGateWeights, dWInput, learningRate);
        DecoderOutputGateWeights = optimizer.UpdateWeights(DecoderOutputGateWeights, dWOutput, learningRate);
        DecoderCellGateWeights = optimizer.UpdateWeights(DecoderCellGateWeights, dWCell, learningRate);
        DecoderForgetGateBiases = optimizer.UpdateBiases(DecoderForgetGateBiases, dbForget, learningRate);
        DecoderInputGateBiases = optimizer.UpdateBiases(DecoderInputGateBiases, dbInput, learningRate);
        DecoderOutputGateBiases = optimizer.UpdateBiases(DecoderOutputGateBiases, dbOutput, learningRate);
        DecoderCellGateBiases = optimizer.UpdateBiases(DecoderCellGateBiases, dbCell, learningRate);
        
        decoderInputCache.Clear();
        decoderForgetCache.Clear();
        decoderInputGateCache.Clear();
        decoderOutputGateCache.Clear();
        decoderCellGateCache.Clear();
        decoderNewCellCache.Clear();
        decoderNewHiddenCache.Clear();
        
        return dHiddenStates.ToArray();
    }
    public TTensor[] EncoderBackward(TTensor[] gradOutputs, double learningRate) 
    {
        int timeSteps = gradOutputs.GetLength(0);
        var dWForget = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWInput = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWOutput = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWCell = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dbForget = tensorOperations.DefaultBiases(HiddenShape);
        var dbInput = tensorOperations.DefaultBiases(HiddenShape);
        var dbOutput = tensorOperations.DefaultBiases(HiddenShape);
        var dbCell = tensorOperations.DefaultBiases(HiddenShape);
        
        TTensor dHiddenNext = tensorOperations.DefaultState(HiddenShape);
        TTensor dCellNext = tensorOperations.DefaultCell(HiddenShape);
        List<TTensor> dHiddenStates = [];

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            TTensor cell = encoderNewCellCache[t];
            TTensor prevCell = t == 0 ? tensorOperations.DefaultCell(HiddenShape) : encoderNewCellCache[t - 1];
            TTensor input = encoderInputCache[t];
            TTensor forget = encoderForgetCache[t];
            TTensor inputGate = encoderInputGateCache[t];
            TTensor outputGate = encoderOutputGateCache[t];
            TTensor cellGate = encoderCellGateCache[t];
            
            TTensor dHidden = tensorOperations.Add(gradOutputs[t], dHiddenNext);
            TTensor tanhCell = TanhActivation.Function(cell);
            TTensor dOutputGate = tensorOperations.Multiply(dHidden, tanhCell);
            TTensor dOutputGatePre = tensorOperations.Multiply(dOutputGate, SigmoidActivation.Derivative(outputGate));
            
            TTensor dCell = tensorOperations.Add(tensorOperations.Multiply(dHidden, outputGate, TanhActivation.Derivative(cell)), dCellNext);

            var dInputGate = tensorOperations.Multiply(dCell, cellGate);
            var dInputGatePre = tensorOperations.Multiply(dInputGate, SigmoidActivation.Derivative(inputGate));

            var dCellCandidate = tensorOperations.Multiply(dCell, inputGate);
            var dCellCandidatePre = tensorOperations.Multiply(dCellCandidate, TanhActivation.Derivative(cellGate));

            var dForgetGate = tensorOperations.Multiply(dCell, prevCell);
            var dForgetGatePre = tensorOperations.Multiply(dForgetGate, SigmoidActivation.Derivative(forget));
            
            TTensor concat = tensorOperations.Concat(input, t == 0 ? tensorOperations.DefaultState(HiddenShape) : encoderNewHiddenCache[t - 1]);
            
            tensorOperations.UpdateAccumulatedWeights(dWForget, dForgetGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWInput, dInputGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWOutput, dOutputGatePre, concat);
            tensorOperations.UpdateAccumulatedWeights(dWCell, dCellCandidatePre, concat);
            tensorOperations.UpdateAccumulatedBiases(dbForget, dForgetGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbInput, dInputGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbOutput, dOutputGatePre);
            tensorOperations.UpdateAccumulatedBiases(dbCell, dCellCandidatePre);

            dCellNext = tensorOperations.Multiply(dCell, forget);
            dHiddenNext = tensorOperations.Add(
                tensorOperations.Multiply(EncoderForgetGateWeights, dForgetGatePre),
                tensorOperations.Multiply(EncoderInputGateWeights, dInputGatePre),
                tensorOperations.Multiply(EncoderOutputGateWeights, dOutputGatePre),
                tensorOperations.Multiply(EncoderCellGateWeights, dCellCandidatePre)
            );
            
            dHiddenStates.Add(dHiddenNext);
        }
        
        EncoderForgetGateWeights = optimizer.UpdateWeights(EncoderForgetGateWeights, dWForget, learningRate);
        EncoderInputGateWeights = optimizer.UpdateWeights(EncoderInputGateWeights, dWInput, learningRate); 
        EncoderOutputGateWeights = optimizer.UpdateWeights(EncoderOutputGateWeights, dWOutput, learningRate);
        EncoderCellGateWeights = optimizer.UpdateWeights(EncoderCellGateWeights, dWCell, learningRate);
        EncoderForgetGateBiases = optimizer.UpdateBiases(EncoderForgetGateBiases, dbForget, learningRate);
        EncoderInputGateBiases = optimizer.UpdateBiases(EncoderInputGateBiases, dbInput, learningRate);
        EncoderOutputGateBiases = optimizer.UpdateBiases(EncoderOutputGateBiases, dbOutput, learningRate);
        EncoderCellGateBiases = optimizer.UpdateBiases(EncoderCellGateBiases, dbCell, learningRate);
        
        encoderInputCache.Clear();
        encoderForgetCache.Clear();
        encoderInputGateCache.Clear();
        encoderOutputGateCache.Clear();
        encoderCellGateCache.Clear();
        encoderNewCellCache.Clear();
        encoderNewHiddenCache.Clear();
        
        return dHiddenStates.ToArray();
    }
    #endregion
    
    public virtual TTensor[] Backward(TTensor[] gradOutputs, double learningRate)
    {
        var dEncoderOutputs = DecoderBackward(gradOutputs, learningRate);
        var dInputs = EncoderBackward(dEncoderOutputs, learningRate);
        return dInputs;
    }

    public void Reset()
    {
        EncoderForgetGateWeights = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderInputGateWeights = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderOutputGateWeights = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderCellGateWeights = tensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderForgetGateBiases = tensorOperations.DefaultBiases(HiddenShape);
        EncoderInputGateBiases = tensorOperations.DefaultBiases(HiddenShape);
        EncoderOutputGateBiases = tensorOperations.DefaultBiases(HiddenShape);
        EncoderCellGateBiases = tensorOperations.DefaultBiases(HiddenShape);
        
        DecoderForgetGateWeights = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderInputGateWeights = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderOutputGateWeights = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderCellGateWeights = tensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderForgetGateBiases = tensorOperations.DefaultBiases(OutputShape);
        DecoderInputGateBiases = tensorOperations.DefaultBiases(OutputShape);
        DecoderOutputGateBiases = tensorOperations.DefaultBiases(OutputShape);
        DecoderCellGateBiases = tensorOperations.DefaultBiases(OutputShape);
        
        encoderInputCache.Clear();
        encoderForgetCache.Clear();
        encoderInputGateCache.Clear();
        encoderOutputGateCache.Clear();
        encoderCellGateCache.Clear();
        encoderNewCellCache.Clear();
        encoderNewHiddenCache.Clear();
        
        decoderInputCache.Clear();
        decoderForgetCache.Clear();
        decoderInputGateCache.Clear();
        decoderOutputGateCache.Clear();
        decoderCellGateCache.Clear();
        decoderNewCellCache.Clear();
        decoderNewHiddenCache.Clear();
    }
}

public interface ILstmTensorOperations<TWeights, TBiases, TTensor>
{
    public TWeights DefaultWeights(int[] outputShape, int[] inputShape);
    public TBiases DefaultBiases(int[] outputShape);
    public TTensor DefaultState(int[] shape);
    public TTensor DefaultCell(int[] shape);
    
    public TTensor Concat(TTensor a, TTensor b);
    public TTensor Multiply(TWeights a, TTensor b);
    public TTensor Add(TTensor a, TBiases b);
    public TTensor Add(TTensor a, TTensor b);
    public TTensor Add(TTensor a, TTensor b, TTensor c, TTensor d);
    public TTensor Multiply(TTensor a, TTensor b);
    public TTensor Multiply(TTensor a, TTensor b, TTensor c);
    
    public void UpdateAccumulatedWeights(TWeights weights, TTensor dForgetGatePre, TTensor concat);
    public void UpdateAccumulatedBiases(TBiases biases, TTensor dForgetGatePre);
}