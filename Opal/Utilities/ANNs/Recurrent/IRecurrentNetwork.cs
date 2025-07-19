namespace Opal.Utilities.ANNs.Recurrent;

public interface IRecurrentNetwork : INueralNetwork
{
    string Name { get; set; }
    List<IRecurrentLayer> Layers { get; set; }
    List<double[]> PredictSequence(List<double[]> inputSequence);
    void Train(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences, int epochs, double learningRate);
    double EvaluateLoss(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences);
}
