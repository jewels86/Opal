namespace Opal.Utilities;

public static class LossFunctions
{
    public static double CrossEntropy(double[] predicted, int actual)
    {
        return -Math.Log(predicted[actual] + 1e-10);
    }

    public static double CrossEntropy(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual arrays must have the same length.");

        double loss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss -= actual[i] * Math.Log(predicted[i] + 1e-10);
        }
        return loss;
    }
    public static double MeanSquaredError(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual arrays must have the same length.");

        double sum = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.Length;
    }

    public static double[] Softmax(double[] predicted)
    {
        double max = predicted.Max();
        double sum = predicted.Sum(p => Math.Exp(p - max));
        return predicted.Select(p => Math.Exp(p - max) / sum).ToArray();
    }
}