namespace Opal.Utilities.ANNs.Recurrent;

public interface IRecurrentLayer : ILayer
{
    public void ResetState();
    public double[][] Forward(double[] input, double[][] previousState);
    public double[][] Backward(double[][] gradOutput, double learningRate);
}