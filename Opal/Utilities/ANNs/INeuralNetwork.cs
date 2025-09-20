namespace Opal.Utilities.ANNs;

public interface INeuralNetwork
{
    public Tensor Predict(Tensor input);
    public void Train(List<Tensor> inputs, List<Tensor> targets, int epochs, double learningRate);
    public double EvaluateLoss(List<Tensor> inputs, List<Tensor> targets);
    public void Reset();
}