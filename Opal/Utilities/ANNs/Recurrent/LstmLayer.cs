namespace Opal.Utilities.ANNs.Recurrent;

public class LstmLayer : IRecurrentLayer
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public double[] HiddenState { get; private set; }
    public double[] CellState { get; private set; }
    public ILayer.ActivationFunction ActivationFunction { get; set; }
    public ILayer.ActivationFunctionDerivative ActivationFunctionDerivative { get; set; }
    
    private double[,] W_xi, W_hi, b_i; // input gate
    private double[,] W_xf, W_hf, b_f; // forget gate
    private double[,] W_xc, W_hc, b_c; // cell gate
    private double[,] W_xo, W_ho, b_o; // output gate

    private List<double[]> inputs, hiddenStates, cellStates;
    private List<double[]> inputGates, forgetGates, outputGates, cellCandidates;

    public LstmLayer(int inputSize, int hiddenSize, ILayer.ActivationFunction activationFunction,
        ILayer.ActivationFunctionDerivative activationFunctionDerivative)
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        
        ActivationFunction = activationFunction;
        ActivationFunctionDerivative = activationFunctionDerivative;
        
        HiddenState = new double[HiddenSize];
        CellState = new double[HiddenSize];
        
        
    }
}