namespace Opal.Utilities.ANNs.Recurrent;

public class RnnLayer : IRecurrentLayer
{
    public double[] State { get; private set; }
    public double[][] Weights { get; private set; }
    public ILayer.ActivationFunction ActivationFunction { get; set; }
    public ILayer.ActivationFunctionDerivative ActivationFunctionDerivative { get; set; }
    public int InputSize { get; private set; }
    public int HiddenSize { get; private set; }

    private double[,] W_x; // hidden size by input size
    private double[,] W_h; // hidden size by hidden size
    private double[] b; // hidden size
    private double[] lastInput;
    private double[] lastPreActivation;
    private List<double[]> inputsPerStep = [];
    private List<double[]> preActivationPerStep = [];
    private List<double[]> statesPerStep = [];
    private double[,] dW_x;
    private double[,] dW_h;
    private double[] db;
    
    public RnnLayer(int inputSize, int hiddenSize, ILayer.ActivationFunction activationFunction, 
        ILayer.ActivationFunctionDerivative activationFunctionDerivative)
    {
        ActivationFunction = activationFunction;
        ActivationFunctionDerivative = activationFunctionDerivative;
        
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        W_x = MathFunctions.RandomMatrix(hiddenSize, inputSize);
        W_h = MathFunctions.RandomMatrix(hiddenSize, hiddenSize);
        b = new double[hiddenSize];
        State = new double[hiddenSize];
        dW_x = new double[hiddenSize, inputSize];
        dW_h = new double[hiddenSize, hiddenSize];
        db = new double[hiddenSize];
    }

    public double[] Forward(double[] input)
    {
        double[] z = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
        {
            double sum = b[i];
            for (int j = 0; j < InputSize; j++)
                sum += W_x[i, j] * input[j];
            for (int j = 0; j < HiddenSize; j++)
                sum += W_h[j, i] * State[j];
            
            z[i] = sum;
        }

        double[] output = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
            output[i] = ActivationFunction(z[i]);
        lastInput = input;
        lastPreActivation = z;
        State = output;
        return output;
    }
    public double[] Backward(double[] gradOutput, double learningRate)
    {
        double[] dz = new double[HiddenSize];
        for (int i = 0; i< HiddenSize; i++) 
            dz[i] = gradOutput[i] * ActivationFunctionDerivative(lastPreActivation[i]);

        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
                W_x[i, j] -= learningRate * dz[i] * lastInput[j];
        }
        
        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < HiddenSize; j++)
                W_h[i, j] -= learningRate * dz[i] * State[j];
        }
        
        for (int i = 0; i < HiddenSize; i++)
            b[i] -= learningRate * dz[i];
        
        double[] gradInput = new double[InputSize];
        for (int i = 0; i < InputSize; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < HiddenSize; j++)
                sum += W_x[j, i] * dz[j];
            gradInput[i] = sum;
        }
        
        double[] gradPrevState = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < HiddenSize; j++)
                sum += W_h[i, j] * dz[j];
            gradPrevState[i] = sum;
        }
        
        return gradInput;
    }

    public void Reset()
    {
        W_x = MathFunctions.RandomMatrix(HiddenSize, InputSize);
        W_h = MathFunctions.RandomMatrix(HiddenSize, HiddenSize);
        b = new double[HiddenSize];
        ResetState();
    }

    public List<double[]> ForwardSequence(List<double[]> inputSequence)
    {
        inputsPerStep.Clear();
        preActivationPerStep.Clear();
        statesPerStep.Clear();
        
        List<double[]> outputs = [];
        double[] currentState = State;
        foreach (var input in inputSequence)
        {
            double[] z = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                double sum = b[i];
                for (int j = 0; j < InputSize; j++)
                    sum += W_x[i, j] * input[j];
                for (int j = 0; j < HiddenSize; j++)
                    sum += W_h[j, i] * currentState[j];
                
                z[i] = sum;
            }
            
            double[] output = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
                output[i] = ActivationFunction(z[i]);
            inputsPerStep.Add(input);
            preActivationPerStep.Add(z);
            statesPerStep.Add(currentState);

            currentState = output;
            outputs.Add(output);
        }

        State = currentState;
        return outputs;
    }

    public List<double[]> BackwardSequence(List<double[]> gradOutputs, double learningRate)
    {
        dW_x = new double[HiddenSize, InputSize];
        dW_h = new double[HiddenSize, HiddenSize];
        db = new double[HiddenSize];
        
        double[] grad_h_next = new double[HiddenSize];
        List<double[]> gradInputsPerStep = [];

        for (int t = gradOutputs.Count - 1; t >= 0; t--)
        {
            var gradOutput = gradOutputs[t];
            var z = preActivationPerStep[t];
            var input = inputsPerStep[t];
            var prevState = statesPerStep[t];

            double[] dz = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
                dz[i] = (gradOutput[i] + grad_h_next[i]) * ActivationFunctionDerivative(z[i]);
            
            double[] gradInput = new double[InputSize];
            for (int i = 0; i < InputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < HiddenSize; j++)
                    sum += W_x[j, i] * dz[j];
                gradInput[i] = sum;
            }
            gradInputsPerStep.Insert(0, gradInput);

            for (int i = 0; i < HiddenSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                    dW_x[i, j] += dz[i] * input[j];
                for (int j = 0; j < HiddenSize; j++)
                    dW_h[i, j] += dz[i] * prevState[j];

                db[i] += dz[i];
            }
            
            double[] gradPrev = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < HiddenSize; j++)
                    sum += W_h[j, i] * dz[j];
                gradPrev[i] = sum;
            }
            grad_h_next = gradPrev;
        }

        for (int i = 0; i < HiddenSize; i++)
        {
            for (int j = 0; j < InputSize; j++) 
                W_x[i, j] -= learningRate * dW_x[i, j];
            for (int j = 0; j < HiddenSize; j++)
                W_h[i, j] -= learningRate * dW_h[i, j];
            b[i] -= learningRate * db[i];
        }

        return gradInputsPerStep;
    }

    public void ResetState()
    {
        State = new double[HiddenSize];
    }
}