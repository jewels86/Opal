using Opal.Utilities.ANNs.Recurrent;

namespace Opal.Utilities.ANNs.Lstm;

using static MathFunctions;
using static BinaryWriting;

public abstract class LstmAttentionLayer<T> where T : LstmAttentionBackpropCache, new()
{
    public string Tag { get; private set; } = "LSTM Attention Layer";
    
    public T BackpropCache { get; set; }
    
    #region Parameters and Functions
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
    public Func<double[], double[]> DTanh { get; set; } = TanhDerivative;
    public Func<double[], double[]> DSigmoid { get; set; } = SigmoidDerivative;
    #endregion
    
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

    #region Encoder, Attention, and Decoder
    #region Encoder
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
    #endregion

    #region Attention
    public delegate void CacheAttentionDelegate(double[] scores, double[] context);
    public double[] Attention(double[,] h, double[] prevState, CacheAttentionDelegate? cache = null, Action<object>? alignmentCacheAction = null)
    {
        int timeSteps = h.GetLength(0);
        int hiddenSize = h.GetLength(1);
        double[] scores = new double[timeSteps];
        for (int j = 0; j < timeSteps; j++)
            scores[j] = Alignment(GetInputFromSample(h, j), prevState, alignmentCacheAction).Sum();
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
    #endregion

    #region Decoder
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
        var (alignmentCache, alignmentCacheAction) = PrepareToCacheAlignment();
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
            double[] context = Attention(encoderHiddenStates, prevHidden, attentionCacheFunc, cache ? alignmentCacheAction : null);
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
            FinalizeAlignmentCache(alignmentCache);
        }

        return ToMatrix2D(hiddenStates);
    }
    #endregion
    #endregion
    
    public double[,] Forward(double[,] input, double[,] output)
    {
        double[,] encoderHiddenStates = Encoder(input);
        double[,] decoderHiddenStates = Decoder(output, encoderHiddenStates);
        return decoderHiddenStates;
    }

    #region Backwards
    public void DecoderBackward(double[,] gradOutputs, double learningRate)
    {
        int timeSteps = BackpropCache.DecoderHiddenStates.GetLength(0) - 1;
        int hiddenSize = HiddenSize;

        double[,] dWf = new double[DecoderForgetGateWeight.GetLength(0), DecoderForgetGateWeight.GetLength(1)];
        double[,] dWi = new double[DecoderInputGateWeight.GetLength(0), DecoderInputGateWeight.GetLength(1)];
        double[,] dWc = new double[DecoderCellGateWeight.GetLength(0), DecoderCellGateWeight.GetLength(1)];
        double[,] dWo = new double[DecoderOutputGateWeight.GetLength(0), DecoderOutputGateWeight.GetLength(1)];
        double[] dBf = new double[hiddenSize];
        double[] dBi = new double[hiddenSize];
        double[] dBc = new double[hiddenSize];
        double[] dBo = new double[hiddenSize];

        double[] dNextHidden = new double[hiddenSize];
        double[] dNextCell = new double[hiddenSize];

        for (int t = timeSteps; t >= 1; t--)
        {
            double[] hidden = GetInputFromSample(BackpropCache.DecoderHiddenStates, t);
            double[] cell = GetInputFromSample(BackpropCache.DecoderCellStates, t);
            double[] prevCell = GetInputFromSample(BackpropCache.DecoderCellStates, t - 1);
            double[] forget = GetInputFromSample(BackpropCache.DecoderForgetGates, t);
            double[] inputGate = GetInputFromSample(BackpropCache.DecoderInputGates, t);
            double[] cellCandidate = GetInputFromSample(BackpropCache.DecoderCellCandidates, t);
            double[] outputGate = GetInputFromSample(BackpropCache.DecoderOutputGates, t);

            double[] dHidden = Add(GetInputFromSample(gradOutputs, t - 1), dNextHidden);

            double[] tanhCell = Tanh(cell);
            double[] dOutputGate = Multiply(dHidden, tanhCell);
            double[] dOutputGatePre = Multiply(dOutputGate, DSigmoid(outputGate));

            double[] dCell = Add(Multiply(dHidden, outputGate, DTanh(cell)), dNextCell);

            double[] dInputGate = Multiply(dCell, cellCandidate);
            double[] dInputGatePre = Multiply(dInputGate, DSigmoid(inputGate));

            double[] dCellCandidate = Multiply(dCell, inputGate);
            double[] dCellCandidatePre = Multiply(dCellCandidate, DTanh(cellCandidate));

            double[] dForgetGate = Multiply(dCell, prevCell);
            double[] dForgetGatePre = Multiply(dForgetGate, DSigmoid(forget));

            double[] combined = GetInputFromSample(BackpropCache.DecoderInputs, t - 1)
                .Concat(GetInputFromSample(BackpropCache.AttentionContextVectors, t - 1))
                .Concat(GetInputFromSample(BackpropCache.DecoderHiddenStates, t - 1)).ToArray();

            AddToMatrix(dWf, OuterProduct(dForgetGatePre, combined));
            AddToMatrix(dWi, OuterProduct(dInputGatePre, combined));
            AddToMatrix(dWc, OuterProduct(dCellCandidatePre, combined));
            AddToMatrix(dWo, OuterProduct(dOutputGatePre, combined));
            AddToVector(dBf, dForgetGatePre);
            AddToVector(dBi, dInputGatePre);
            AddToVector(dBc, dCellCandidatePre);
            AddToVector(dBo, dOutputGatePre);

            dNextCell = Multiply(dCell, forget);

            int offset = BackpropCache.DecoderInputs.GetLength(1) + BackpropCache.AttentionContextVectors.GetLength(1);
            int prevHiddenSize = hiddenSize;
            dNextHidden = new double[hiddenSize];

            for (int h = 0; h < hiddenSize; h++)
            {
                for (int k = 0; k < prevHiddenSize; k++)
                {
                    dNextHidden[k] +=
                        dForgetGatePre[h] * DecoderForgetGateWeight[offset + k, h] +
                        dInputGatePre[h] * DecoderInputGateWeight[offset + k, h] +
                        dCellCandidatePre[h] * DecoderCellGateWeight[offset + k, h] +
                        dOutputGatePre[h] * DecoderOutputGateWeight[offset + k, h];
                }
            }
        }

        SubtractInPlace(DecoderForgetGateWeight, Multiply(dWf, learningRate));
        SubtractInPlace(DecoderInputGateWeight, Multiply(dWi, learningRate));
        SubtractInPlace(DecoderCellGateWeight, Multiply(dWc, learningRate));
        SubtractInPlace(DecoderOutputGateWeight, Multiply(dWo, learningRate));
        SubtractInPlace(DecoderForgetGateBias, Multiply(dBf, learningRate));
        SubtractInPlace(DecoderInputGateBias, Multiply(dBi, learningRate));
        SubtractInPlace(DecoderCellGateBias, Multiply(dBc, learningRate));
        SubtractInPlace(DecoderOutputGateBias, Multiply(dBo, learningRate));
    }

    public double[,] AttentionBackward(double[,] gradOutputs, double learningRate)
    {
        int timeSteps = BackpropCache.AttentionContextVectors.GetLength(0);
        int hiddenSize = BackpropCache.AttentionContextVectors.GetLength(1);

        double[,] gradEncoderHidden = new double[BackpropCache.EncoderHiddenStates.GetLength(0), BackpropCache.EncoderHiddenStates.GetLength(1)];

        for (int t = 0; t < timeSteps; t++)
        {
            double[] scores = GetInputFromSample(BackpropCache.AttentionScores, t);
            double[] attentionWeights = Softmax(scores);

            double[] gradContext = GetInputFromSample(gradOutputs, t);

            for (int j = 0; j < BackpropCache.EncoderHiddenStates.GetLength(0); j++)
            {
                double[] encoderHidden = GetInputFromSample(BackpropCache.EncoderHiddenStates, j);
                for (int k = 0; k < hiddenSize; k++)
                {
                    gradEncoderHidden[j, k] += attentionWeights[j] * gradContext[k];
                }
            }

            double[] gradAttentionWeights = new double[attentionWeights.Length];
            for (int j = 0; j < attentionWeights.Length; j++)
            {
                double[] encoderHidden = GetInputFromSample(BackpropCache.EncoderHiddenStates, j);
                gradAttentionWeights[j] = Dot(gradContext, encoderHidden)[0];
            }

            double[] gradScores = new double[attentionWeights.Length];
            for (int j = 0; j < attentionWeights.Length; j++)
            {
                double sum = 0.0;
                for (int l = 0; l < attentionWeights.Length; l++)
                {
                    double delta = (j == l) ? 1.0 : 0.0;
                    sum += gradAttentionWeights[l] * attentionWeights[j] * (delta - attentionWeights[l]);
                }
                gradScores[j] = sum;
            }

            TrainAlignment(BackpropCache, t, gradScores, learningRate);
        }

        return gradEncoderHidden;
    }

    public void EncoderBackward(double[,] gradOutputs, double learningRate)
    {
        int timeSteps = BackpropCache.EncoderHiddenStates.GetLength(0) - 1;
        int hiddenSize = HiddenSize;

        double[,] dWf = new double[ForgetGateWeight.GetLength(0), ForgetGateWeight.GetLength(1)];
        double[,] dWi = new double[InputGateWeight.GetLength(0), InputGateWeight.GetLength(1)];
        double[,] dWc = new double[CellGateWeight.GetLength(0), CellGateWeight.GetLength(1)];
        double[,] dWo = new double[OutputGateWeight.GetLength(0), OutputGateWeight.GetLength(1)];
        double[] dBf = new double[hiddenSize];
        double[] dBi = new double[hiddenSize];
        double[] dBc = new double[hiddenSize];
        double[] dBo = new double[hiddenSize];

        double[] dNextHidden = new double[hiddenSize];
        double[] dNextCell = new double[hiddenSize];

        for (int t = timeSteps; t >= 1; t--)
        {
            double[] hidden = GetInputFromSample(BackpropCache.EncoderHiddenStates, t);
            double[] cell = GetInputFromSample(BackpropCache.EncoderCellStates, t);
            double[] prevCell = GetInputFromSample(BackpropCache.EncoderCellStates, t - 1);
            double[] forget = GetInputFromSample(BackpropCache.EncoderForgetGates, t);
            double[] inputGate = GetInputFromSample(BackpropCache.EncoderInputGates, t);
            double[] cellCandidate = GetInputFromSample(BackpropCache.EncoderCellCandidates, t);
            double[] outputGate = GetInputFromSample(BackpropCache.EncoderOutputGates, t);

            double[] dHidden = Add(GetInputFromSample(gradOutputs, t - 1), dNextHidden);

            double[] tanhCell = Tanh(cell);
            double[] dOutputGate = Multiply(dHidden, tanhCell);
            double[] dOutputGatePre = Multiply(dOutputGate, DSigmoid(outputGate));

            double[] dCell = Add(Multiply(dHidden, outputGate, DTanh(cell)), dNextCell);

            double[] dInputGate = Multiply(dCell, cellCandidate);
            double[] dInputGatePre = Multiply(dInputGate, DSigmoid(inputGate));

            double[] dCellCandidate = Multiply(dCell, inputGate);
            double[] dCellCandidatePre = Multiply(dCellCandidate, DTanh(cellCandidate));

            double[] dForgetGate = Multiply(dCell, prevCell);
            double[] dForgetGatePre = Multiply(dForgetGate, DSigmoid(forget));

            double[] combined = GetInputFromSample(BackpropCache.EncoderInputs, t - 1)
                .Concat(GetInputFromSample(BackpropCache.EncoderHiddenStates, t - 1)).ToArray();

            AddToMatrix(dWf, OuterProduct(dForgetGatePre, combined));
            AddToMatrix(dWi, OuterProduct(dInputGatePre, combined));
            AddToMatrix(dWc, OuterProduct(dCellCandidatePre, combined));
            AddToMatrix(dWo, OuterProduct(dOutputGatePre, combined));
            AddToVector(dBf, dForgetGatePre);
            AddToVector(dBi, dInputGatePre);
            AddToVector(dBc, dCellCandidatePre);
            AddToVector(dBo, dOutputGatePre);

            dNextCell = Multiply(dCell, forget);

            int offset = BackpropCache.EncoderInputs.GetLength(1);
            int prevHiddenSize = hiddenSize;
            dNextHidden = new double[hiddenSize];

            for (int h = 0; h < hiddenSize; h++)
            {
                for (int k = 0; k < prevHiddenSize; k++)
                {
                    dNextHidden[k] +=
                        dForgetGatePre[h] * ForgetGateWeight[offset + k, h] +
                        dInputGatePre[h] * InputGateWeight[offset + k, h] +
                        dCellCandidatePre[h] * CellGateWeight[offset + k, h] +
                        dOutputGatePre[h] * OutputGateWeight[offset + k, h];
                }
            }
        }

        SubtractInPlace(ForgetGateWeight, Multiply(dWf, learningRate));
        SubtractInPlace(InputGateWeight, Multiply(dWi, learningRate));
        SubtractInPlace(CellGateWeight, Multiply(dWc, learningRate));
        SubtractInPlace(OutputGateWeight, Multiply(dWo, learningRate));
        SubtractInPlace(ForgetGateBias, Multiply(dBf, learningRate));
        SubtractInPlace(InputGateBias, Multiply(dBi, learningRate));
        SubtractInPlace(CellGateBias, Multiply(dBc, learningRate));
        SubtractInPlace(OutputGateBias, Multiply(dBo, learningRate));
    }
    #endregion
    
    public void Backward(double[,] gradOutputs, double learningRate)
    {
        DecoderBackward(gradOutputs, learningRate);
        double[,] gradEncoderHidden = AttentionBackward(gradOutputs, learningRate);
        EncoderBackward(gradEncoderHidden, learningRate);
    }

    public abstract void ResetAlignment();
    public abstract void SaveAlignment(BinaryWriter writer);
    public abstract void LoadAlignment(BinaryReader reader);
    public abstract double[] Alignment(double[] encoderHidden, double[] decoderHidden, Action<object>? alignmentCacheAction = null);
    public abstract (Dictionary<string, object>, Action<object>) PrepareToCacheAlignment();
    public abstract void FinalizeAlignmentCache(Dictionary<string, object> alignmentCache);
    public abstract void TrainAlignment(LstmAttentionBackpropCache cache, int decoderTimeStep, double[] gradScores, double learningRate);
    
    public void Save(BinaryWriter writer)
    {
        writer.Write(Tag);
        writer.Write(InputSize);
        writer.Write(HiddenSize);
        writer.Write(OutputSize);
        
        WriteMatrix(writer, ForgetGateWeight);
        WriteMatrix(writer, InputGateWeight);
        WriteMatrix(writer, CellGateWeight);
        WriteMatrix(writer, OutputGateWeight);
        
        WriteVector(writer, ForgetGateBias);
        WriteVector(writer, InputGateBias);
        WriteVector(writer, CellGateBias);
        WriteVector(writer, OutputGateBias);
        
        WriteMatrix(writer, DecoderForgetGateWeight);
        WriteMatrix(writer, DecoderInputGateWeight);
        WriteMatrix(writer, DecoderCellGateWeight);
        WriteMatrix(writer, DecoderOutputGateWeight);
        
        WriteVector(writer, DecoderForgetGateBias);
        WriteVector(writer, DecoderInputGateBias);
        WriteVector(writer, DecoderCellGateBias);
        WriteVector(writer, DecoderOutputGateBias);

        SaveAlignment(writer);
    }
    
    public static TLayer Load<TLayer, TCache>(BinaryReader reader, TLayer layer) where TLayer : LstmAttentionLayer<TCache> where TCache : LstmAttentionBackpropCache, new()
    {
        string tag = reader.ReadString();
        int inputSize = reader.ReadInt32();
        int hiddenSize = reader.ReadInt32();
        int outputSize = reader.ReadInt32();
        
        layer.ForgetGateWeight = ReadMatrix(reader);
        layer.InputGateWeight = ReadMatrix(reader);
        layer.CellGateWeight = ReadMatrix(reader);
        layer.OutputGateWeight = ReadMatrix(reader);
        
        layer.ForgetGateBias = ReadVector(reader);
        layer.InputGateBias = ReadVector(reader);
        layer.CellGateBias = ReadVector(reader);
        layer.OutputGateBias = ReadVector(reader);
        
        layer.DecoderForgetGateWeight = ReadMatrix(reader);
        layer.DecoderInputGateWeight = ReadMatrix(reader);
        layer.DecoderCellGateWeight = ReadMatrix(reader);
        layer.DecoderOutputGateWeight = ReadMatrix(reader);
        
        layer.DecoderForgetGateBias = ReadVector(reader);
        layer.DecoderInputGateBias = ReadVector(reader);
        layer.DecoderCellGateBias = ReadVector(reader);
        layer.DecoderOutputGateBias = ReadVector(reader);

        layer.LoadAlignment(reader);

        return layer;
    }

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
        ResetAlignment();
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