namespace Opal.Utilities.ANNs;

public class FfNetwork : INueralNetwork
{
    public List<FfLayer> Layers { get; set; }
    public string Name { get; set; }
    
    public FfNetwork(string name)
    {
        Name = name;
        Layers = [];
    }
    
    public void AddLayer(FfLayer layer) => Layers.Add(layer);

    public double[] Predict(double[] input)
    {
        foreach (var layer in Layers)
            input = layer.Forward(input);
        return input;
    }

    public void Train(double[][] inputs, double[][] targets, int epochs, double lr)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var predicted = Predict(inputs[i]);
                var actual = targets[i];
                totalLoss += LossFunctions.CrossEntropy(predicted, actual);
                var grad = predicted.Zip(actual, (p, t) => p - t).ToArray();
                for (int l = Layers.Count - 1; l >= 0; l--)
                    grad = Layers[l].Backward(grad, lr);
            }
            Core.Log(Name, 3, $"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / inputs.Length}");
        }
    }

    public double EvaluateLoss(double[][] inputs, double[][] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var predicted = Predict(inputs[i]);
            var actual = targets[i];
            totalLoss += LossFunctions.CrossEntropy(predicted, actual);
        }
        return totalLoss / inputs.Length;
    }
    public void Reset()
    {
        foreach (var layer in Layers) layer.Reset();
    }
}