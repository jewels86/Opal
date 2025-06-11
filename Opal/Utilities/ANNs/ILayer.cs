namespace Opal.Utilities.ANNs;

public interface ILayer
{
    public double[] Forward(double[] input);
    public double[] Backward(double[] gradOutput, double learningRate);
    public void Reset();
}