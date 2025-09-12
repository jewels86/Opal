namespace Opal.Utilities.ANNs.Recurrent;

using static MathFunctions;
using static Logging;
using System.IO;
using static BinaryWriting;

public abstract class LstmAttentionLayer
{
    public string Tag { get; set; } = "LSTM Attention Layer";
    
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
    public abstract Func<double[], double[], double[]> AlignmentFunction { get; set; }
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
    }

    public (double[] hidden, double[] cell) Encoder(double[] input, double[] prevHidden, double[] prevCell)
    {
        double[] combined = input.Concat(prevHidden).ToArray();
        
        double[] forget = SigmoidActivation(Add(Multiply(ForgetGateWeight, combined), ForgetGateBias));
        double[] inputGate = SigmoidActivation(Add(Multiply(InputGateWeight, combined), InputGateBias));
        double[] cellCandidate = TanhActivation(Add(Multiply(CellGateWeight, combined), CellGateBias));
            
        double[] cell = Add(Multiply(forget, prevCell), Multiply(inputGate, cellCandidate));
        double[] outputGate = SigmoidActivation(Add(Multiply(OutputGateWeight, combined), OutputGateBias));
        double[] hidden = Multiply(outputGate, TanhActivation(cell));
            
        return (hidden, cell);
    }

    public double[,] Encoder(double[,] x)
    {
        List<double[]> hiddenStates = [new double[HiddenSize]];
        List<double[]> cellStates = [new double[HiddenSize]];
        
        int timeSteps = x.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            double[] input = GetInputFromSample(x, t);
            var (hidden, cell) = Encoder(input, hiddenStates.Last(), cellStates.Last());
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }

        return ToMatrix2D(hiddenStates);
    }

    public double[] Attention(double[,] h, double[] prevState)
    {
        int timeSteps = h.GetLength(0);
        int hiddenSize = h.GetLength(1);

        double[] scores = new double[timeSteps];
        for (int j = 0; j < timeSteps; j++)
            scores[j] = AlignmentFunction(GetInputFromSample(h, j), prevState).Sum();

        double[] attentionWeights = Softmax(scores);

        double[] context = new double[hiddenSize];
        for (int j = 0; j < timeSteps; j++)
        {
            double[] h_j = GetInputFromSample(h, j);
            for (int k = 0; k < hiddenSize; k++)
                context[k] += attentionWeights[j] * h_j[k];
        }
        return context;
    }

    public (double[] hidden, double[] cell) Decoder(double[] prevOutput, double[] context, double[] prevHidden, double[] prevCell)
    {
        double[] combined = prevOutput.Concat(context).Concat(prevHidden).ToArray();

        double[] forget = SigmoidActivation(Add(Multiply(DecoderForgetGateWeight, combined), DecoderForgetGateBias));
        double[] inputGate = SigmoidActivation(Add(Multiply(DecoderInputGateWeight, combined), DecoderInputGateBias));
        double[] cellCandidate = TanhActivation(Add(Multiply(DecoderCellGateWeight, combined), DecoderCellGateBias));

        double[] cell = Add(Multiply(forget, prevCell), Multiply(inputGate, cellCandidate));
        double[] outputGate = SigmoidActivation(Add(Multiply(DecoderOutputGateWeight, combined), DecoderOutputGateBias));
        double[] hidden = Multiply(outputGate, TanhActivation(cell));

        return (hidden, cell);
    }
    
    public double[,] Decoder(double[,] y, double[,] encoderHiddenStates)
    {
        List<double[]> hiddenStates = [new double[HiddenSize]];
        List<double[]> cellStates = [new double[HiddenSize]];

        int timeSteps = y.GetLength(0);
        for (int t = 0; t < timeSteps; t++)
        {
            double[] prevOutput = GetInputFromSample(y, t);
            double[] prevHidden = hiddenStates.Last();
            double[] prevCell = cellStates.Last();

            double[] context = Attention(encoderHiddenStates, prevHidden);

            var (hidden, cell) = Decoder(prevOutput, context, prevHidden, prevCell);
            hiddenStates.Add(hidden);
            cellStates.Add(cell);
        }

        return ToMatrix2D(hiddenStates);
    }
    
    // TODO: Abstract method for training attention
}