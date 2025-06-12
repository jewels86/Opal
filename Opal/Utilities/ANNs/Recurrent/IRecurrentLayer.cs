namespace Opal.Utilities.ANNs.Recurrent;

public interface IRecurrentLayer : ILayer
{
    public void ResetState();
    public List<double[]> ForwardSequence(List<double[]> inputSequence);
    public List<double[]> BackwardSequence(List<double[]> gradOutputs, double learningRate);
}