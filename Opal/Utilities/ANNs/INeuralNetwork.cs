namespace Opal.Utilities.ANNs;

/// <summary>
/// Represents a generic neural network interface providing methods for prediction,
/// training, evaluation, and resetting the network state.
/// </summary>
/// <typeparam name="T">The data type used by the neural network, which must be a non-nullable type.</typeparam>
public interface INeuralNetwork<T> where T : notnull
{
    public T Predict(T input);
    public void Train(T[] inputs, T[] targets, int epochs, double learningRate);
    public double EvaluateLoss(T[] inputs, T[] targets);
    public void Reset();
}