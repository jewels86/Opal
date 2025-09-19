namespace Opal.Utilities.ANNs.Recurrent;

public interface IRecurrentNetwork
{
    public string Name { get; }
    public List<IRecurrentLayer> Layers { get; }
    public List<double[]> PredictSequence(double[,] inputSequence);
    public void Train(double[,,] inputSequences, double[,,] targetSequences, int epochs, double learningRate);
    public double EvaluateLoss(double[,,] inputSequences, double[,,] targetSequences);
}
