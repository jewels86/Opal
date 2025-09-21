using static Opal.Mathematics.Tensors;

namespace Opal.Utilities.ANNs;

public class FfLayer<T> : ILayer<T>
{
    public double[,] Weights { get; set; }
    public double[] Biases { get; set; }
    public int InputSize { get; }
    public int OutputSize { get; }
    public Func<double, double> Activation { get; }
    public Func<double, double> ActivationDerivative { get; }

    private double[] lastInput = [];
    private double[] lastZ = [];

    public FfLayer(int inputSize, int outputSize, Func<double, double> activation,
        Func<double, double> activationDerivative)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = activation;
        ActivationDerivative = activationDerivative;
        
        Weights = RandomMatrix(OutputSize, InputSize);
        Biases = new double[OutputSize];
    }

    public double[] Forward(double[] input)
    {
        double[] z = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            z[i] = Biases[i];
            for (int j = 0; j < InputSize; j++)
                z[i] += Weights[i, j] * input[j];
            z[i] = Activation(z[i]);
        }
        return z;
    }

    public double[] Backward(double[] gradOutput, double learningRate)
    {
        double[] gradInput = new double[InputSize];
        double[] gradZ = new double[OutputSize];
        
        for (int i = 0; i < OutputSize; i++)
            gradZ[i] = gradOutput[i] * ActivationDerivative(lastZ[i]);

        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                gradInput[j] += gradZ[i] * Weights[i, j];
                Weights[i, j] -= learningRate * gradZ[i] * lastInput[j];
            }
            Biases[i] -= learningRate * gradZ[i];
        }
        return gradInput;
    }
    
    public void Reset()
    {
        Weights = RandomMatrix(OutputSize, InputSize);
        Biases = new double[OutputSize];
        lastInput = [];
        lastZ = [];
    }
}