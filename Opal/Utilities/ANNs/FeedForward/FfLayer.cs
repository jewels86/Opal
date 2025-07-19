namespace Opal.Utilities.ANNs;

public class FfLayer : ILayer
{
    public double[,] Weights { get; set; }
    public double[] Biases { get; set; }
    public int InputSize, OutputSize, N;

    public double[] Input, Z, Activation = [];
    public Func<double[], double[]> ActivationFunction;
    public Func<double[], double[]> ActivationFunctionDerivative;

    public FfLayer(int inputSize, int neuronCounts, Func<double[], double[]> activationFunction,
        Func<double[], double[]> activationFunctionDerivative)
    {
        InputSize = inputSize;
        OutputSize = neuronCounts;
        ActivationFunction = activationFunction;
        ActivationFunctionDerivative = activationFunctionDerivative;
        
        Weights = new double[OutputSize, InputSize];
        Biases = new double[OutputSize];

        var rand = new Random();
        for (int i = 0; i < OutputSize; i++) 
            for (int j = 0; j < InputSize; j++)
                Weights[i, j] = rand.NextDouble() * 2 - 1;
        N = inputSize;
    }

    public double[] Forward(double[] input)
    {
        Input = input;
        Z = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            Z[i] = Biases[i];
            for (int j = 0; j < InputSize; j++)
                Z[i] += Weights[i, j] * input[i];
        }
        Activation = ActivationFunction(Z);
        return Activation;
    }

    public double[] Backward(double[] gradOutput, double learningRate = 0.01)
    {
        var gradZ = new double[OutputSize];
        var gradInput = new double[InputSize];
        
        var dz = ActivationFunctionDerivative(Z);
        for (int i = 0; i < OutputSize; i++)
            gradZ[i] = gradOutput[i] * dz[i];
        
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                gradInput[j] += Weights[i, j] * gradZ[i];
                Weights[i, j] -= learningRate * gradZ[i] * Input[i];
            }
            Biases[i] -= learningRate * gradZ[i];
        }

        return gradInput;
    }
    
    public void Reset()
    {
        Input = new double[InputSize];
        Z = new double[OutputSize];
        Activation = new double[OutputSize];
        Biases = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
            for (int j = 0; j < InputSize; j++)
                Weights[i, j] = 0.0;
    }
}