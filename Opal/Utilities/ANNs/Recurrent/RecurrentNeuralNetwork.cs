namespace Opal.Utilities.ANNs.Recurrent;

public class RecurrentNeuralNetwork : IRecurrentNetwork
{
    public string Name { get; set; }
    public List<IRecurrentLayer> Layers { get; set; }

    public RecurrentNeuralNetwork(string name)
    {
        Name = name;
        Layers = [];
    }

    public void AddLayer(IRecurrentLayer layer) => Layers.Add(layer);

    public List<double[]> PredictSequence(List<double[]> inputSequence)
    {
        var sequence = inputSequence;
        foreach (var layer in Layers)
            sequence = layer.ForwardSequence(sequence);
        return sequence;
    }

    public void Train(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputSequences.Count; i++)
            {
                var predicted = PredictSequence(inputSequences[i]);
                var actual = targetSequences[i];
                // Assume last layer output and target have same length
                double seqLoss = 0.0;
                for (int t = 0; t < predicted.Count; t++)
                    seqLoss += LossFunctions.CrossEntropy(predicted[t], actual[t]);
                totalLoss += seqLoss / predicted.Count;

                // Compute gradient for output layer
                var grad = new List<double[]>();
                for (int t = 0; t < predicted.Count; t++)
                    grad.Add(predicted[t].Zip(actual[t], (p, a) => p - a).ToArray());

                // Backpropagate through layers in reverse
                for (int l = Layers.Count - 1; l >= 0; l--)
                    grad = Layers[l].BackwardSequence(grad, learningRate);
            }
            Core.Log(Name, 3, $"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / inputSequences.Count}");
        }
    }

    public double EvaluateLoss(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputSequences.Count; i++)
        {
            var predicted = PredictSequence(inputSequences[i]);
            var actual = targetSequences[i];
            double seqLoss = 0.0;
            for (int t = 0; t < predicted.Count; t++)
                seqLoss += LossFunctions.CrossEntropy(predicted[t], actual[t]);
            totalLoss += seqLoss / predicted.Count;
        }
        return totalLoss / inputSequences.Count;
    }

    public void Reset()
    {
        foreach (var layer in Layers)
            layer.Reset();
    }


    public double[] Predict(double[] input)
    {
        // Treat as a sequence of length 1
        var outputSeq = PredictSequence(new List<double[]> { input });
        return outputSeq.Count > 0 ? outputSeq[0] : Array.Empty<double>();
    }

    public void Train(double[][] inputs, double[][] targets, int epochs, double learningRate)
    {
        var inputSeqs = inputs.Select(x => new List<double[]> { x }).ToList();
        var targetSeqs = targets.Select(x => new List<double[]> { x }).ToList();
        Train(inputSeqs, targetSeqs, epochs, learningRate);
    }

    public double EvaluateLoss(double[][] inputs, double[][] targets)
    {
        var inputSeqs = inputs.Select(x => new List<double[]> { x }).ToList();
        var targetSeqs = targets.Select(x => new List<double[]> { x }).ToList();
        return EvaluateLoss(inputSeqs, targetSeqs);
    }
}