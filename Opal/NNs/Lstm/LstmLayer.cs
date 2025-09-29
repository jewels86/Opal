using Opal.Mathematics;

namespace Opal.NNs.Lstm;

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
    
    protected List<TTensor> EncoderInputCache = [];
    protected List<TTensor> EncoderForgetCache = [];
    protected List<TTensor> EncoderInputGateCache = [];
    protected List<TTensor> EncoderOutputGateCache = [];
    protected List<TTensor> EncoderCellGateCache = [];
    protected List<TTensor> EncoderNewCellCache = [];
    protected List<TTensor> EncoderNewHiddenCache = [];
    
    protected List<TTensor> DecoderInputCache = [];
    protected List<TTensor> DecoderForgetCache = [];
    protected List<TTensor> DecoderInputGateCache = [];
    protected List<TTensor> DecoderOutputGateCache = [];
    protected List<TTensor> DecoderCellGateCache = [];
    protected List<TTensor> DecoderNewCellCache = [];
    protected List<TTensor> DecoderNewHiddenCache = [];

    protected readonly ActivationFunction<TTensor> SigmoidActivation;
    protected readonly ActivationFunction<TTensor> TanhActivation;

    protected readonly ILstmTensorOperations<TWeights, TBiases, TTensor> TensorOperations;
    protected readonly IOptimizer<TWeights, TBiases> Optimizer;

    public LstmLayer(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation)
    {
        InputShape = inputShape;
        HiddenShape = hiddenShape;
        OutputShape = outputShape;
        
        this.TensorOperations = tensorOperations;
        this.Optimizer = optimizer;
        
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
        TTensor concat = TensorOperations.Concat(input, hidden);
        TTensor forget = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(EncoderForgetGateWeights, concat), EncoderForgetGateBiases));
        TTensor inputGate = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(EncoderInputGateWeights, concat), EncoderInputGateBiases));
        TTensor outputGate = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(EncoderOutputGateWeights, concat), EncoderOutputGateBiases));
        TTensor cellGate = TanhActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(EncoderCellGateWeights, concat), EncoderCellGateBiases));
        TTensor newCell = TensorOperations.Add(TensorOperations.Multiply(forget, cell), TensorOperations.Multiply(inputGate, cellGate));
        TTensor newHidden = TensorOperations.Multiply(outputGate, TanhActivation.Function(newCell));
        if (cacheHandler is not null)
            cacheHandler(input, forget, inputGate, outputGate, cellGate, newCell, newHidden);
        return (newHidden, newCell);
    }

    public TTensor[] Encoder(TTensor[] inputs, bool cache = true)
    {
        List<TTensor> hiddenStates = [TensorOperations.DefaultState(HiddenShape)];
        List<TTensor> cellStates = [TensorOperations.DefaultCell(OutputShape)];
        
        EncoderCacheHandler? cacheFunc = !cache ? null : (input, forget, inputGate, outputGate, cellGate, newCell, newHidden) =>
        {
            EncoderInputCache.Add(input);
            EncoderForgetCache.Add(forget);
            EncoderInputGateCache.Add(inputGate);
            EncoderOutputGateCache.Add(outputGate);
            EncoderCellGateCache.Add(cellGate);
            EncoderNewCellCache.Add(newCell);
            EncoderNewHiddenCache.Add(newHidden);
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
        TTensor concat = TensorOperations.Concat(input, hidden);
        TTensor forget = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(DecoderForgetGateWeights, concat), DecoderForgetGateBiases));
        TTensor inputGate = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(DecoderInputGateWeights, concat), DecoderInputGateBiases));
        TTensor outputGate = SigmoidActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(DecoderOutputGateWeights, concat), DecoderOutputGateBiases));
        TTensor cellGate = TanhActivation.Function(TensorOperations.Add(
            TensorOperations.Multiply(DecoderCellGateWeights, concat), DecoderCellGateBiases));
        TTensor newCell = TensorOperations.Add(TensorOperations.Multiply(forget, cell), TensorOperations.Multiply(inputGate, cellGate));
        TTensor newHidden = TensorOperations.Multiply(outputGate, TanhActivation.Function(newCell));
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
            DecoderInputCache.Add(input);
            DecoderForgetCache.Add(forget);
            DecoderInputGateCache.Add(inputGate);
            DecoderOutputGateCache.Add(outputGate);
            DecoderCellGateCache.Add(cellGate);
            DecoderNewCellCache.Add(newCell);
            DecoderNewHiddenCache.Add(newHidden);
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
    
    public TTensor[] Forward(TTensor[] inputs, bool cache) => Forward(inputs, TensorOperations.DefaultState(HiddenShape), TensorOperations.DefaultCell(OutputShape), cache);
    public TTensor[] Forward(TTensor[] inputs) => Forward(inputs, true);
    
    #region Backward
    public TTensor[] DecoderBackward(TTensor[] gradOutputs, double learningRate)
    {
        int timeSteps = gradOutputs.GetLength(0);
        var dWForget = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWInput = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWOutput = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dWCell = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        var dbForget = TensorOperations.DefaultBiases(OutputShape);
        var dbInput = TensorOperations.DefaultBiases(OutputShape);
        var dbOutput = TensorOperations.DefaultBiases(OutputShape);
        var dbCell = TensorOperations.DefaultBiases(OutputShape);
        
        TTensor dHiddenNext = TensorOperations.DefaultState(OutputShape);
        TTensor dCellNext = TensorOperations.DefaultCell(OutputShape);
        List<TTensor> dHiddenStates = [];

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            TTensor cell = DecoderNewCellCache[t];
            TTensor prevCell = t == 0 ? TensorOperations.DefaultCell(OutputShape) : DecoderNewCellCache[t - 1];
            TTensor input = DecoderInputCache[t];
            TTensor forget = DecoderForgetCache[t];
            TTensor inputGate = DecoderInputGateCache[t];
            TTensor outputGate = DecoderOutputGateCache[t];
            TTensor cellGate = DecoderCellGateCache[t];
            
            TTensor dHidden = TensorOperations.Add(gradOutputs[t], dHiddenNext);
            TTensor tanhCell = TanhActivation.Function(cell);
            TTensor dOutputGate = TensorOperations.Multiply(dHidden, tanhCell);
            TTensor dOutputGatePre = TensorOperations.Multiply(dOutputGate, SigmoidActivation.Derivative(outputGate));
            
            TTensor dCell = TensorOperations.Add(TensorOperations.Multiply(dHidden, outputGate, TanhActivation.Derivative(cell)), dCellNext);

            var dInputGate = TensorOperations.Multiply(dCell, cellGate);
            var dInputGatePre = TensorOperations.Multiply(dInputGate, SigmoidActivation.Derivative(inputGate));

            var dCellCandidate = TensorOperations.Multiply(dCell, inputGate);
            var dCellCandidatePre = TensorOperations.Multiply(dCellCandidate, TanhActivation.Derivative(cellGate));

            var dForgetGate = TensorOperations.Multiply(dCell, prevCell);
            var dForgetGatePre = TensorOperations.Multiply(dForgetGate, SigmoidActivation.Derivative(forget));
            
            TTensor concat = TensorOperations.Concat(input, t == 0 ? TensorOperations.DefaultState(HiddenShape) : DecoderNewHiddenCache[t - 1]);
            
            TensorOperations.UpdateAccumulatedWeights(dWForget, dForgetGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWInput, dInputGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWOutput, dOutputGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWCell, dCellCandidatePre, concat);
            TensorOperations.UpdateAccumulatedBiases(dbForget, dForgetGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbInput, dInputGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbOutput, dOutputGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbCell, dCellCandidatePre);

            dCellNext = TensorOperations.Multiply(dCell, forget);
            dHiddenNext = TensorOperations.Add(
                TensorOperations.Multiply(DecoderForgetGateWeights, dForgetGatePre),
                TensorOperations.Multiply(DecoderInputGateWeights, dInputGatePre),
                TensorOperations.Multiply(DecoderOutputGateWeights, dOutputGatePre),
                TensorOperations.Multiply(DecoderCellGateWeights, dCellCandidatePre)
            );
            
            dHiddenStates.Add(dHiddenNext);
        }
        
        DecoderForgetGateWeights = Optimizer.UpdateWeights(DecoderForgetGateWeights, dWForget, learningRate);
        DecoderInputGateWeights = Optimizer.UpdateWeights(DecoderInputGateWeights, dWInput, learningRate);
        DecoderOutputGateWeights = Optimizer.UpdateWeights(DecoderOutputGateWeights, dWOutput, learningRate);
        DecoderCellGateWeights = Optimizer.UpdateWeights(DecoderCellGateWeights, dWCell, learningRate);
        DecoderForgetGateBiases = Optimizer.UpdateBiases(DecoderForgetGateBiases, dbForget, learningRate);
        DecoderInputGateBiases = Optimizer.UpdateBiases(DecoderInputGateBiases, dbInput, learningRate);
        DecoderOutputGateBiases = Optimizer.UpdateBiases(DecoderOutputGateBiases, dbOutput, learningRate);
        DecoderCellGateBiases = Optimizer.UpdateBiases(DecoderCellGateBiases, dbCell, learningRate);
        
        DecoderInputCache.Clear();
        DecoderForgetCache.Clear();
        DecoderInputGateCache.Clear();
        DecoderOutputGateCache.Clear();
        DecoderCellGateCache.Clear();
        DecoderNewCellCache.Clear();
        DecoderNewHiddenCache.Clear();
        
        return dHiddenStates.ToArray();
    }
    public TTensor[] EncoderBackward(TTensor[] gradOutputs, double learningRate) 
    {
        int timeSteps = gradOutputs.GetLength(0);
        var dWForget = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWInput = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWOutput = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dWCell = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        var dbForget = TensorOperations.DefaultBiases(HiddenShape);
        var dbInput = TensorOperations.DefaultBiases(HiddenShape);
        var dbOutput = TensorOperations.DefaultBiases(HiddenShape);
        var dbCell = TensorOperations.DefaultBiases(HiddenShape);
        
        TTensor dHiddenNext = TensorOperations.DefaultState(HiddenShape);
        TTensor dCellNext = TensorOperations.DefaultCell(HiddenShape);
        List<TTensor> dHiddenStates = [];

        for (int t = timeSteps - 1; t >= 0; t--)
        {
            TTensor cell = EncoderNewCellCache[t];
            TTensor prevCell = t == 0 ? TensorOperations.DefaultCell(HiddenShape) : EncoderNewCellCache[t - 1];
            TTensor input = EncoderInputCache[t];
            TTensor forget = EncoderForgetCache[t];
            TTensor inputGate = EncoderInputGateCache[t];
            TTensor outputGate = EncoderOutputGateCache[t];
            TTensor cellGate = EncoderCellGateCache[t];
            
            TTensor dHidden = TensorOperations.Add(gradOutputs[t], dHiddenNext);
            TTensor tanhCell = TanhActivation.Function(cell);
            TTensor dOutputGate = TensorOperations.Multiply(dHidden, tanhCell);
            TTensor dOutputGatePre = TensorOperations.Multiply(dOutputGate, SigmoidActivation.Derivative(outputGate));
            
            TTensor dCell = TensorOperations.Add(TensorOperations.Multiply(dHidden, outputGate, TanhActivation.Derivative(cell)), dCellNext);

            var dInputGate = TensorOperations.Multiply(dCell, cellGate);
            var dInputGatePre = TensorOperations.Multiply(dInputGate, SigmoidActivation.Derivative(inputGate));

            var dCellCandidate = TensorOperations.Multiply(dCell, inputGate);
            var dCellCandidatePre = TensorOperations.Multiply(dCellCandidate, TanhActivation.Derivative(cellGate));

            var dForgetGate = TensorOperations.Multiply(dCell, prevCell);
            var dForgetGatePre = TensorOperations.Multiply(dForgetGate, SigmoidActivation.Derivative(forget));
            
            TTensor concat = TensorOperations.Concat(input, t == 0 ? TensorOperations.DefaultState(HiddenShape) : EncoderNewHiddenCache[t - 1]);
            
            TensorOperations.UpdateAccumulatedWeights(dWForget, dForgetGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWInput, dInputGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWOutput, dOutputGatePre, concat);
            TensorOperations.UpdateAccumulatedWeights(dWCell, dCellCandidatePre, concat);
            TensorOperations.UpdateAccumulatedBiases(dbForget, dForgetGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbInput, dInputGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbOutput, dOutputGatePre);
            TensorOperations.UpdateAccumulatedBiases(dbCell, dCellCandidatePre);

            dCellNext = TensorOperations.Multiply(dCell, forget);
            dHiddenNext = TensorOperations.Add(
                TensorOperations.Multiply(EncoderForgetGateWeights, dForgetGatePre),
                TensorOperations.Multiply(EncoderInputGateWeights, dInputGatePre),
                TensorOperations.Multiply(EncoderOutputGateWeights, dOutputGatePre),
                TensorOperations.Multiply(EncoderCellGateWeights, dCellCandidatePre)
            );
            
            dHiddenStates.Add(dHiddenNext);
        }
        
        EncoderForgetGateWeights = Optimizer.UpdateWeights(EncoderForgetGateWeights, dWForget, learningRate);
        EncoderInputGateWeights = Optimizer.UpdateWeights(EncoderInputGateWeights, dWInput, learningRate); 
        EncoderOutputGateWeights = Optimizer.UpdateWeights(EncoderOutputGateWeights, dWOutput, learningRate);
        EncoderCellGateWeights = Optimizer.UpdateWeights(EncoderCellGateWeights, dWCell, learningRate);
        EncoderForgetGateBiases = Optimizer.UpdateBiases(EncoderForgetGateBiases, dbForget, learningRate);
        EncoderInputGateBiases = Optimizer.UpdateBiases(EncoderInputGateBiases, dbInput, learningRate);
        EncoderOutputGateBiases = Optimizer.UpdateBiases(EncoderOutputGateBiases, dbOutput, learningRate);
        EncoderCellGateBiases = Optimizer.UpdateBiases(EncoderCellGateBiases, dbCell, learningRate);
        
        EncoderInputCache.Clear();
        EncoderForgetCache.Clear();
        EncoderInputGateCache.Clear();
        EncoderOutputGateCache.Clear();
        EncoderCellGateCache.Clear();
        EncoderNewCellCache.Clear();
        EncoderNewHiddenCache.Clear();
        
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
        EncoderForgetGateWeights = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderInputGateWeights = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderOutputGateWeights = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderCellGateWeights = TensorOperations.DefaultWeights(HiddenShape, InputShape);
        EncoderForgetGateBiases = TensorOperations.DefaultBiases(HiddenShape);
        EncoderInputGateBiases = TensorOperations.DefaultBiases(HiddenShape);
        EncoderOutputGateBiases = TensorOperations.DefaultBiases(HiddenShape);
        EncoderCellGateBiases = TensorOperations.DefaultBiases(HiddenShape);
        
        DecoderForgetGateWeights = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderInputGateWeights = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderOutputGateWeights = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderCellGateWeights = TensorOperations.DefaultWeights(OutputShape, HiddenShape);
        DecoderForgetGateBiases = TensorOperations.DefaultBiases(OutputShape);
        DecoderInputGateBiases = TensorOperations.DefaultBiases(OutputShape);
        DecoderOutputGateBiases = TensorOperations.DefaultBiases(OutputShape);
        DecoderCellGateBiases = TensorOperations.DefaultBiases(OutputShape);
        
        EncoderInputCache.Clear();
        EncoderForgetCache.Clear();
        EncoderInputGateCache.Clear();
        EncoderOutputGateCache.Clear();
        EncoderCellGateCache.Clear();
        EncoderNewCellCache.Clear();
        EncoderNewHiddenCache.Clear();
        
        DecoderInputCache.Clear();
        DecoderForgetCache.Clear();
        DecoderInputGateCache.Clear();
        DecoderOutputGateCache.Clear();
        DecoderCellGateCache.Clear();
        DecoderNewCellCache.Clear();
        DecoderNewHiddenCache.Clear();
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