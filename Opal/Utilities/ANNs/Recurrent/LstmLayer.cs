namespace Opal.Utilities.ANNs.Recurrent;

using static MathFunctions;

public class LstmLayer
{
    public ILayer.ActivationFunction SigmoidFunction { get; set; } = Sigmoid;
    public ILayer.ActivationFunction TanhFunction { get; set; } = Tanh;

    public int InputSize { get; private set; }
    public int HiddenSize { get; private set; }
    public int BatchSize { get; private set; }
    
    public List<double[]> CellStates { get; private set; } = [];
    public List<double[]> HiddenStates { get; private set; } = [];
    public List<double[]> InputStates { get; private set; } = [];
    
    public double[,] ForgetGateWeight { get; set; }
    public double[,] InputGateWeight { get; set; }
    public double[,] CellStateWeight { get; set; }
    public double[,] OutputGateWeight { get; set; }
    
    public double[] ForgetGateBias { get; set; }
    public double[] InputGateBias { get; set; }
    public double[] CellStateBias { get; set; } 
    public double[] OutputGateBias { get; set; }

    private List<double[]> _forgetGates = [];
    private List<double[]> _inputGates = [];
    private List<double[]> _cellCandidates = [];
    private List<double[]> _outputGates = [];
    private List<double[]> _concatInputs = [];
    private List<List<double[]>> _cellStateTimeline = [];
    private List<List<double[]>> _hiddenStateTimeline = [];

    public LstmLayer(int inputSize, int hiddenSize, int batchSize)
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        BatchSize = batchSize;
        
        ForgetGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        InputGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        CellStateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);
        OutputGateWeight = RandomMatrix(inputSize + hiddenSize, hiddenSize);

        ForgetGateBias = new double[HiddenSize];
        InputGateBias = new double[HiddenSize];
        CellStateBias = new double[HiddenSize];
        OutputGateBias = new double[HiddenSize];
    }

    public double[,,] Forward(double[,,] inputSequence, bool reset = true)
    {
        if (reset) ResetState();
        int batch = inputSequence.GetLength(0);
        int time = inputSequence.GetLength(1);
        int inputSize = inputSequence.GetLength(2);

        double[,,] output = new double[batch, time, HiddenSize];

        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < time; j++)
            {
                double[] input = GetInputFromSample(GetBatchSample(inputSequence, i), j);
                double[] concat = input.Concat(HiddenStates[i]).ToArray();

                double[] forgetGate = Apply(SigmoidFunction, Add(Multiply(ForgetGateWeight, concat), ForgetGateBias));
                double[] inputGate = Apply(SigmoidFunction, Add(Multiply(InputGateWeight, concat), InputGateBias));
                double[] cellCandidate = Apply(TanhFunction, Add(Multiply(CellStateWeight, concat), CellStateBias));

                double[] newCellState = Add(
                    Multiply(forgetGate, CellStates[i]),
                    Multiply(inputGate, cellCandidate));
                double[] outputGate = Apply(SigmoidFunction, Add(Multiply(OutputGateWeight, concat), OutputGateBias));
                double[] hiddenState = Multiply(outputGate, Apply(TanhFunction, newCellState));

                CellStates[i] = newCellState;
                HiddenStates[i] = hiddenState;
                InputStates[i] = input;

                for (int h = 0; h < HiddenSize; h++)
                    output[i, j, h] = hiddenState[h];
            }
        }
        return output;
    }

    public void ResetState()
    {
        CellStates.Clear();
        HiddenStates.Clear();
        InputStates.Clear();
        
        for (int i = 0; i < BatchSize; i++)
        {
            CellStates.Add(new double[HiddenSize]);
            HiddenStates.Add(new double[HiddenSize]);
            InputStates.Add(new double[InputSize]);
        }
    }
}