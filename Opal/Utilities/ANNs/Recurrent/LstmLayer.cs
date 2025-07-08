namespace Opal.Utilities.ANNs.Recurrent;

using static MathFunctions;
using static Logging;

public class LstmLayer
{
    public string Tag { get; set; } = "LSTM Layer?";
    
    public ILayer.ActivationFunction SigmoidFunction { get; set; } = Sigmoid;
    public ILayer.ActivationFunction TanhFunction { get; set; } = Tanh;
    public ILayer.ActivationFunctionDerivative SigmoidDerivativeFunction { get; set; } = SigmoidDerivative;
    public ILayer.ActivationFunctionDerivative TanhDerivativeFunction { get; set; } = TanhDerivative;

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
    
    private double[,] dForgetGateWeight;
    private double[,] dInputGateWeight;
    private double[,] dCellStateWeight;
    private double[,] dOutputGateWeight;
    private double[] dForgetGateBias;
    private double[] dInputGateBias;
    private double[] dCellStateBias;
    private double[] dOutputGateBias;

    // Add cell state timeline for correct backprop
    private List<double[]> _cellStatesPerTimestep = [];

    public LstmLayer(int inputSize, int hiddenSize, int batchSize, string? tag = null)
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
        
        dForgetGateWeight = ZeroMatrix(inputSize + hiddenSize, hiddenSize);
        dInputGateWeight = ZeroMatrix(inputSize + hiddenSize, hiddenSize);
        dCellStateWeight = ZeroMatrix(inputSize + hiddenSize, hiddenSize);
        dOutputGateWeight = ZeroMatrix(inputSize + hiddenSize, hiddenSize);
        dForgetGateBias = new double[HiddenSize];
        dInputGateBias = new double[HiddenSize];
        dCellStateBias = new double[HiddenSize];
        dOutputGateBias = new double[HiddenSize];

        Tag = tag ?? Tag;
    }

    public double[,,] Forward(double[,,] inputSequence, bool reset = true)
    {
        if (reset)
        {
            ResetState();
            _forgetGates.Clear();
            _inputGates.Clear();
            _cellCandidates.Clear();
            _outputGates.Clear();
            _concatInputs.Clear();
            _cellStatesPerTimestep.Clear(); // clear cell state timeline
        }
        int batch = inputSequence.GetLength(0);
        int time = inputSequence.GetLength(1);
        int inputSize = inputSequence.GetLength(2);

        double[,,] output = new double[batch, time, HiddenSize];

        for (int t = 0; t < time; t++)
        {
            _forgetGates.Add(new double[batch * HiddenSize]);
            _inputGates.Add(new double[batch * HiddenSize]);
            _cellCandidates.Add(new double[batch * HiddenSize]);
            _outputGates.Add(new double[batch * HiddenSize]);
            _concatInputs.Add(new double[batch * (inputSize + HiddenSize)]);
        }

        for (int i = 0; i < batch; i++)
        {
            Core.Log(Tag, (int)LogLevel.Debug, $"Processing batch {i + 1}/{batch}...");
            for (int j = 0; j < time; j++)
            {
                Core.Log(Tag, (int)LogLevel.Debug, $"Processing timestep {j + 1}/{time} for batch {i + 1}/{batch}...");
                double[] input = GetInputFromSample(GetBatchSample(inputSequence, i), j);
                double[] concat = input.Concat(HiddenStates[i]).ToArray();

                double[] forgetGate = Apply(SigmoidFunction, Add(Multiply(concat, ForgetGateWeight), ForgetGateBias));
                double[] inputGate = Apply(SigmoidFunction, Add(Multiply(concat, InputGateWeight), InputGateBias));
                double[] cellCandidate = Apply(TanhFunction, Add(Multiply(concat, CellStateWeight), CellStateBias));

                double[] newCellState = Add(
                    Multiply(forgetGate, CellStates[i]),
                    Multiply(inputGate, cellCandidate));
                double[] outputGate = Apply(SigmoidFunction, Add(Multiply(concat, OutputGateWeight), OutputGateBias));
                double[] hiddenState = Multiply(outputGate, Apply(TanhFunction, newCellState));

                CellStates[i] = newCellState;
                HiddenStates[i] = hiddenState;
                InputStates[i] = input;

                Array.Copy(forgetGate, 0, _forgetGates[j], i * HiddenSize, HiddenSize);
                Array.Copy(inputGate, 0, _inputGates[j], i * HiddenSize, HiddenSize);
                Array.Copy(cellCandidate, 0, _cellCandidates[j], i * HiddenSize, HiddenSize);
                Array.Copy(outputGate, 0, _outputGates[j], i * HiddenSize, HiddenSize);
                Array.Copy(concat, 0, _concatInputs[j], i * (inputSize + HiddenSize), inputSize + HiddenSize);

                // Cache the actual cell state for this timestep and batch
                if (_cellStatesPerTimestep.Count <= j)
                    _cellStatesPerTimestep.Add(new double[batch * HiddenSize]);
                Array.Copy(newCellState, 0, _cellStatesPerTimestep[j], i * HiddenSize, HiddenSize);

                for (int h = 0; h < HiddenSize; h++)
                    output[i, j, h] = hiddenState[h];
            }
        }
        return output;
    }

    public double[,,] Backward(double[,,] gradOutput, double learningRate)
    {
        // TODO: Rewrite manually
        int batch = gradOutput.GetLength(0);
        int time = gradOutput.GetLength(1);
        int inputSize = InputSize;
        int hiddenSize = HiddenSize;
        int norm = batch * time;
        double clipValue = 5.0; // You can parameterize this if needed

        // Initialize gradients
        dForgetGateWeight = ZeroMatrix(ForgetGateWeight.GetLength(0), ForgetGateWeight.GetLength(1));
        dInputGateWeight = ZeroMatrix(InputGateWeight.GetLength(0), InputGateWeight.GetLength(1));
        dCellStateWeight = ZeroMatrix(CellStateWeight.GetLength(0), CellStateWeight.GetLength(1));
        dOutputGateWeight = ZeroMatrix(OutputGateWeight.GetLength(0), OutputGateWeight.GetLength(1));
        dForgetGateBias = new double[ForgetGateBias.Length];
        dInputGateBias = new double[InputGateBias.Length];
        dCellStateBias = new double[CellStateBias.Length];
        dOutputGateBias = new double[OutputGateBias.Length];

        // For returning input gradients if needed
        double[,,] gradInput = new double[batch, time, inputSize];

        double[][] dhNext = new double[batch][];
        double[][] dcNext = new double[batch][];
        for (int i = 0; i < batch; i++)
        {
            dhNext[i] = new double[hiddenSize];
            dcNext[i] = new double[hiddenSize];
        }

        // Backward through time
        for (int t = time - 1; t >= 0; t--)
        {
            for (int b = 0; b < batch; b++)
            {
                // Indices for caches
                int gateOffset = b * hiddenSize;
                int concatOffset = b * (inputSize + hiddenSize);

                // Retrieve cached values
                double[] forgetGate = new double[hiddenSize];
                double[] inputGate = new double[hiddenSize];
                double[] cellCandidate = new double[hiddenSize];
                double[] outputGate = new double[hiddenSize];
                double[] concat = new double[inputSize + hiddenSize];

                Array.Copy(_forgetGates[t], gateOffset, forgetGate, 0, hiddenSize);
                Array.Copy(_inputGates[t], gateOffset, inputGate, 0, hiddenSize);
                Array.Copy(_cellCandidates[t], gateOffset, cellCandidate, 0, hiddenSize);
                Array.Copy(_outputGates[t], gateOffset, outputGate, 0, hiddenSize);
                Array.Copy(_concatInputs[t], concatOffset, concat, 0, inputSize + HiddenSize);

                // Retrieve cell state and previous cell state from timeline
                double[] cellState = new double[hiddenSize];
                Array.Copy(_cellStatesPerTimestep[t], gateOffset, cellState, 0, hiddenSize);
                double[] prevCellState = t > 0
                    ? new double[hiddenSize]
                    : ZeroVector(hiddenSize);
                if (t > 0)
                    Array.Copy(_cellStatesPerTimestep[t - 1], gateOffset, prevCellState, 0, hiddenSize);

                // Gradients from output
                double[] dH = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dH[h] = gradOutput[b, t, h] + dhNext[b][h];

                // Output gate grad (fix: use tanh(cellState[h]))
                double[] dOutputGate = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dOutputGate[h] = dH[h] * TanhFunction(cellState[h]) * outputGate[h] * (1 - outputGate[h]);

                double[] tanhCell = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    tanhCell[h] = TanhFunction(cellState[h]);

                double[] dCell = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dCell[h] = dH[h] * outputGate[h] * (1 - tanhCell[h] * tanhCell[h]) + dcNext[b][h];

                // Forget gate grad
                double[] dForgetGate = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dForgetGate[h] = dCell[h] * prevCellState[h] * forgetGate[h] * (1 - forgetGate[h]);

                // Input gate grad
                double[] dInputGate = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dInputGate[h] = dCell[h] * cellCandidate[h] * inputGate[h] * (1 - inputGate[h]);

                // Cell candidate grad
                double[] dCellCandidate = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                    dCellCandidate[h] = dCell[h] * inputGate[h] * (1 - cellCandidate[h] * cellCandidate[h]);

                // Gradients w.r.t weights and biases
                AddOuterProduct(dForgetGateWeight, concat, dForgetGate);
                AddOuterProduct(dInputGateWeight, concat, dInputGate);
                AddOuterProduct(dCellStateWeight, concat, dCellCandidate);
                AddOuterProduct(dOutputGateWeight, concat, dOutputGate);

                for (int h = 0; h < hiddenSize; h++)
                {
                    dForgetGateBias[h] += dForgetGate[h];
                    dInputGateBias[h] += dInputGate[h];
                    dCellStateBias[h] += dCellCandidate[h];
                    dOutputGateBias[h] += dOutputGate[h];
                }

                // Gradients w.r.t concat input
                double[] dConcat = new double[inputSize + hiddenSize];
                AddMatVecMul(dConcat, ForgetGateWeight, dForgetGate);
                AddMatVecMul(dConcat, InputGateWeight, dInputGate);
                AddMatVecMul(dConcat, CellStateWeight, dCellCandidate);
                AddMatVecMul(dConcat, OutputGateWeight, dOutputGate);

                // Split dConcat into input and hidden
                for (int h = 0; h < hiddenSize; h++)
                    dhNext[b][h] = dConcat[inputSize + h];
                for (int inp = 0; inp < inputSize; inp++)
                    gradInput[b, t, inp] = dConcat[inp];

                // Gradients w.r.t previous cell state
                for (int h = 0; h < hiddenSize; h++)
                    dcNext[b][h] = dCell[h] * forgetGate[h];
            }
        }

        // Normalize gradients
        DivideInPlace(dForgetGateWeight, norm);
        DivideInPlace(dInputGateWeight, norm);
        DivideInPlace(dCellStateWeight, norm);
        DivideInPlace(dOutputGateWeight, norm);
        DivideInPlace(dForgetGateBias, norm);
        DivideInPlace(dInputGateBias, norm);
        DivideInPlace(dCellStateBias, norm);
        DivideInPlace(dOutputGateBias, norm);

        // Clip gradients
        ClipInPlace(dForgetGateWeight, -clipValue, clipValue);
        ClipInPlace(dInputGateWeight, -clipValue, clipValue);
        ClipInPlace(dCellStateWeight, -clipValue, clipValue);
        ClipInPlace(dOutputGateWeight, -clipValue, clipValue);
        ClipInPlace(dForgetGateBias, -clipValue, clipValue);
        ClipInPlace(dInputGateBias, -clipValue, clipValue);
        ClipInPlace(dCellStateBias, -clipValue, clipValue);
        ClipInPlace(dOutputGateBias, -clipValue, clipValue);

        SubtractInPlace(ForgetGateWeight, dForgetGateWeight, learningRate);
        SubtractInPlace(InputGateWeight, dInputGateWeight, learningRate);
        SubtractInPlace(CellStateWeight, dCellStateWeight, learningRate);
        SubtractInPlace(OutputGateWeight, dOutputGateWeight, learningRate);
        SubtractInPlace(ForgetGateBias, dForgetGateBias, learningRate);
        SubtractInPlace(InputGateBias, dInputGateBias, learningRate);
        SubtractInPlace(CellStateBias, dCellStateBias, learningRate);
        SubtractInPlace(OutputGateBias, dOutputGateBias, learningRate);

        return gradInput;
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

    public double[] GetLastHiddenState(int batchIndex) => HiddenStates[batchIndex];
    public double[] GetLastCellState(int batchIndex) => CellStates[batchIndex];
}