namespace Opal.Utilities.ANNs.Recurrent;

using static MathFunctions;
using System.IO;

public abstract class LstmAttentionLayer<T> where T : LstmAttentionBackpropCache, new()
{
    public string Tag { get; private set; } = "LSTM Attention Layer";
    
    public T BackpropCache { get; set; }
    
    public double[,] ForgetGateWeight { get; set; }
    public double[,] InputGateWeight { get; set; }
    public double[,] CellGateWeight { get; set; }
    public double[,] OutputGateWeight { get; set; }
    
    public double[] ForgetGateBias { get; set; }
    public double[] InputGateBias { get; set; }
    public double[] CellGateBias { get; set; }
    public double[] OutputGateBias { get; set; }
    
    public double[,] DecoderForgetGateWeight { get; set; }
    public double[,] DecoderInputGateWeight { get; set; }
    public double[,] DecoderCellGateWeight { get; set; }
    public double[,] DecoderOutputGateWeight { get; set; }
    
    public double[] DecoderForgetGateBias { get; set; }
    public double[] DecoderInputGateBias { get; set; }
    public double[] DecoderCellGateBias { get; set; }
    public double[] DecoderOutputGateBias { get; set; }
    
    public Func<double[], double[]> TanhActivation { get; set; } = Tanh;
    public Func<double[], double[]> SigmoidActivation { get; set; } = Sigmoid;
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }

    public LstmAttentionLayer(int inputSize, int hiddenSize, int outputSize, string? tag = null)
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        OutputSize = outputSize;
        Tag = tag ?? Tag;
        
        ForgetGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        InputGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        CellGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        OutputGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);

        ForgetGateBias = new double[HiddenSize];
        InputGateBias = new double[HiddenSize];
        CellGateBias = new double[HiddenSize];
        OutputGateBias = new double[HiddenSize];
        
        DecoderForgetGateWeight = RandomMatrix(outputSize + hiddenSize + hiddenSize, hiddenSize);
        DecoderInputGateWeight = RandomMatrix(outputSize + hiddenSize + hiddenSize, hiddenSize);
        DecoderCellGateWeight = RandomMatrix(outputSize + hiddenSize + hiddenSize, hiddenSize);
        DecoderOutputGateWeight = RandomMatrix(outputSize + hiddenSize + hiddenSize, hiddenSize);
        
        DecoderForgetGateBias = new double[HiddenSize];
        DecoderInputGateBias = new double[HiddenSize];
        DecoderCellGateBias = new double[HiddenSize];
        DecoderOutputGateBias = new double[HiddenSize];

        BackpropCache = new T();
    }

    public delegate void CacheEncoderDelegate(double[] forget, double[] inputGate, double[] cellCandidate, double[] outputGate);
    public (double[] hidden, double[] cell) Encoder(double[] input, double[] prevHidden, double[] prevCell, CacheEncoderDelegate? cache = null)
    {
        double[] combined = input.Concat(prevHidden).ToArray();
        double[] forget = SigmoidActivation(Add(Multiply(ForgetGateWeight, combined), ForgetGateBias));
        double[] inputGate = SigmoidActivation(Add(Multiply(InputGateWeight, combined), InputGateBias));
        double[] cellCandidate = TanhActivation(Add(Multiply(CellGateWeight, combined), CellGateBias));
        double[] cell = Add(Multiply(forget, prevCell), Multiply(inputGate, cellCandidate));
        double[] outputGate = SigmoidActivation(Add(Multiply(OutputGateWeight, combined), OutputGateBias));
        double[] hidden = Multiply(outputGate, TanhActivation(cell));
        if (cache is not null)
            cache(forget, inputGate, cellCandidate, outputGate);
        return (hidden, cell);
    }

    public double[,] Encoder(double[,] x, bool cache = true)
    {
        List<double[]> hiddenStates = [new double[HiddenSize]];
        List<double[]> cellStates = [new double[HiddenSize]];
        CacheEncoderDelegate? cacheFunc = null;
        List<double[]>? forgetGates = null;
        List<double[]>? inputGates = null;
        List<double[]>? cellCandidates = null;
        List<double[]>? outputGates = null;
        if (cache)
        {
            forgetGates = [];
            inputGates = [];
            cellCandidates = [];
            outputGates = [];
            cacheFunc = (forget, inputGate, cellCandidate, outputGate) =>
            {
                forgetGates.Add(forget);
                inputGates.Add(inputGate);
                cellCandidates.Add(cellCandidate);
                outputGates.Add(outputGate);
            };
        }
        int timeSteps = x.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            double[] input = GetInputFromSample(x, t);
            var (hidden, cell) = Encoder(input, hiddenStates.Last(), cellStates.Last(), cacheFunc);
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }
        if (cache)
        {
            BackpropCache.EncoderInputs = x;
            BackpropCache.EncoderHiddenStates = ToMatrix2D(hiddenStates);
            BackpropCache.EncoderCellStates = ToMatrix2D(cellStates);
            BackpropCache.EncoderForgetGates = ToMatrix2D(forgetGates!);
            BackpropCache.EncoderInputGates = ToMatrix2D(inputGates!);
            BackpropCache.EncoderCellCandidates = ToMatrix2D(cellCandidates!);
            BackpropCache.EncoderOutputGates = ToMatrix2D(outputGates!);
        }
        return ToMatrix2D(hiddenStates);
    }

    public delegate void CacheAttentionDelegate(double[] scores, double[] context);
    public double[] Attention(double[,] h, double[] prevState, CacheAttentionDelegate? cache = null)
    {
        int timeSteps = h.GetLength(0);
        int hiddenSize = h.GetLength(1);
        double[] scores = new double[timeSteps];
        for (int j = 0; j < timeSteps; j++)
            scores[j] = Alignment(GetInputFromSample(h, j), prevState).Sum();
        double[] attentionWeights = Softmax(scores);
        double[] context = new double[hiddenSize];
        for (int j = 0; j < timeSteps; j++)
        {
            double[] hJ = GetInputFromSample(h, j);
            for (int k = 0; k < hiddenSize; k++)
                context[k] += attentionWeights[j] * hJ[k];
        }
        if (cache is not null)
            cache(scores, context);
        return context;
    }

    public delegate void CacheDecoderDelegate(double[] forget, double[] inputGate, double[] cellCandidate, double[] outputGate);
    public (double[] hidden, double[] cell) Decoder(double[] prevOutput, double[] context, double[] prevHidden, double[] prevCell, CacheDecoderDelegate? cache = null)
    {
        double[] combined = prevOutput.Concat(context).Concat(prevHidden).ToArray();
        double[] forget = SigmoidActivation(Add(Multiply(DecoderForgetGateWeight, combined), DecoderForgetGateBias));
        double[] inputGate = SigmoidActivation(Add(Multiply(DecoderInputGateWeight, combined), DecoderInputGateBias));
        double[] cellCandidate = TanhActivation(Add(Multiply(DecoderCellGateWeight, combined), DecoderCellGateBias));
        double[] cell = Add(Multiply(forget, prevCell), Multiply(inputGate, cellCandidate));
        double[] outputGate = SigmoidActivation(Add(Multiply(DecoderOutputGateWeight, combined), DecoderOutputGateBias));
        double[] hidden = Multiply(outputGate, TanhActivation(cell));
        if (cache is not null)
            cache(forget, inputGate, cellCandidate, outputGate);
        return (hidden, cell);
    }

    public double[,] Decoder(double[,] y, double[,] encoderHiddenStates, bool cache = true)
    {
        List<double[]> hiddenStates = [new double[HiddenSize]];
        List<double[]> cellStates = [new double[HiddenSize]];
        List<double[]>? forgetGates = null;
        List<double[]>? inputGates = null;
        List<double[]>? cellCandidates = null;
        List<double[]>? outputGates = null;
        List<double[]>? attentionScores = null;
        List<double[]>? attentionContexts = null;
        CacheDecoderDelegate? cacheFunc = null;
        CacheAttentionDelegate? attentionCacheFunc = null;
        if (cache)
        {
            forgetGates = [];
            inputGates = [];
            cellCandidates = [];
            outputGates = [];
            attentionScores = [];
            attentionContexts = [];
            cacheFunc = (forget, inputGate, cellCandidate, outputGate) =>
            {
                forgetGates.Add(forget);
                inputGates.Add(inputGate);
                cellCandidates.Add(cellCandidate);
                outputGates.Add(outputGate);
            };
            attentionCacheFunc = (scores, context) =>
            {
                attentionScores.Add(scores);
                attentionContexts.Add(context);
            };
        }
        int timeSteps = y.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            double[] prevOutput = GetInputFromSample(y, t);
            double[] prevHidden = hiddenStates.Last();
            double[] prevCell = cellStates.Last();
            double[] context = Attention(encoderHiddenStates, prevHidden, attentionCacheFunc);
            var (hidden, cell) = Decoder(prevOutput, context, prevHidden, prevCell, cacheFunc);
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }
        if (cache)
        {
            BackpropCache.DecoderInputs = y;
            BackpropCache.DecoderHiddenStates = ToMatrix2D(hiddenStates);
            BackpropCache.DecoderCellStates = ToMatrix2D(cellStates);
            BackpropCache.DecoderForgetGates = ToMatrix2D(forgetGates!);
            BackpropCache.DecoderInputGates = ToMatrix2D(inputGates!);
            BackpropCache.DecoderCellCandidates = ToMatrix2D(cellCandidates!);
            BackpropCache.DecoderOutputGates = ToMatrix2D(outputGates!);
            BackpropCache.AttentionScores = ToMatrix2D(attentionScores!);
            BackpropCache.AttentionContextVectors = ToMatrix2D(attentionContexts!);
        }

        return ToMatrix2D(hiddenStates);
    }
    
    public double[,] Forward(double[,] input, double[,] output)
    {
        double[,] encoderHiddenStates = Encoder(input);
        double[,] decoderHiddenStates = Decoder(output, encoderHiddenStates);
        return decoderHiddenStates;
    }
    
    // TODO: Abstract method for training attention

    public abstract void ResetAttention();
    public abstract void SaveAttention(BinaryWriter writer);
    public abstract void LoadAttention(BinaryReader reader);
    public abstract double[] Alignment(double[] encoderHidden, double[] decoderHidden);

    public void Reset()
    {
        ForgetGateWeight = RandomMatrix(InputSize + HiddenSize, HiddenSize);
        InputGateWeight = RandomMatrix(InputSize + HiddenSize, HiddenSize);
        CellGateWeight = RandomMatrix(InputSize + HiddenSize, HiddenSize);
        OutputGateWeight = RandomMatrix(InputSize + HiddenSize, HiddenSize);
        ForgetGateBias = new double[HiddenSize];
        InputGateBias = new double[HiddenSize];
        CellGateBias = new double[HiddenSize];
        OutputGateBias = new double[HiddenSize];
        DecoderForgetGateWeight = RandomMatrix(OutputSize + HiddenSize + HiddenSize, HiddenSize);
        DecoderInputGateWeight = RandomMatrix(OutputSize + HiddenSize + HiddenSize, HiddenSize);
        DecoderCellGateWeight = RandomMatrix(OutputSize + HiddenSize + HiddenSize, HiddenSize);
        DecoderOutputGateWeight = RandomMatrix(OutputSize + HiddenSize + HiddenSize, HiddenSize);
        DecoderForgetGateBias = new double[HiddenSize];
        DecoderInputGateBias = new double[HiddenSize];
        DecoderCellGateBias = new double[HiddenSize];
        DecoderOutputGateBias = new double[HiddenSize];
        ResetAttention();
    }
}

public abstract class LstmAttentionBackpropCache
{
    public double[,] EncoderInputs { get; set; } = new double[0, 0];
    public double[,] EncoderHiddenStates { get; set; } = new double[0, 0];
    public double[,] EncoderCellStates { get; set; } = new double[0, 0];
    public double[,] EncoderForgetGates { get; set; } = new double[0, 0];
    public double[,] EncoderInputGates { get; set; } = new double[0, 0];
    public double[,] EncoderCellCandidates { get; set; } = new double[0, 0];
    public double[,] EncoderOutputGates { get; set; } = new double[0, 0];
        
    public double[,] AttentionScores { get; set; } = new double[0, 0];
    public double[,] AttentionContextVectors { get; set; } = new double[0, 0];
        
    public double[,] DecoderInputs { get; set; } = new double[0, 0];
    public double[,] DecoderHiddenStates { get; set; } = new double[0, 0];
    public double[,] DecoderCellStates { get; set; } = new double[0, 0];
    public double[,] DecoderForgetGates { get; set; } = new double[0, 0];
    public double[,] DecoderInputGates { get; set; } = new double[0, 0];
    public double[,] DecoderCellCandidates { get; set; } = new double[0, 0];
    public double[,] DecoderOutputGates { get; set; } = new double[0, 0];
}