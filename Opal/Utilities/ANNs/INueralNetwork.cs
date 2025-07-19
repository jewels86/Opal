namespace Opal.Utilities.ANNs;

public interface INueralNetwork
{
    public double[] Predict(double[] input);
    public void Train(double[][] inputs, double[][] targets, int epochs, double learningRate);
    public double EvaluateLoss(double[][] inputs, double[][] targets);
    public void Reset();
}