namespace Opal.Utilities.ANNs.Recurrent;

public class LstmNetwork
{
    public string Name { get; set; }
    public List<LstmLayer> Layers { get; set; }

    public LstmNetwork(string name)
    {
        Name = name;
        Layers = [];
    }

    public void AddLayer(LstmLayer layer) => Layers.Add(layer);

    public List<double[]> PredictSequence(List<double[]> inputSequence)
    {
        // Error handling
        if (inputSequence.Select(x => x.Length).Distinct().Count() != 1)
        {
            Core.Log(Name, 2, "Input sequence must have consistent input size.");
            return [];
        }
        if (inputSequence.Count == 0)
        {
            Core.Log(Name, 2, "Input sequence is empty.");
            return [];
        }
        // Convert inputSequence to [batch, time, inputSize] with batch=1
        int time = inputSequence.Count;
        int inputSize = inputSequence[0].Length;
        double[,,] input = new double[1, time, inputSize];
        for (int t = 0; t < time; t++)
            for (int i = 0; i < inputSize; i++)
                input[0, t, i] = inputSequence[t][i];

        double[,,] output = input;
        foreach (var layer in Layers)
            output = layer.Forward(output);

        // Convert output [1, time, hiddenSize] to List<double[]>
        int hiddenSize = output.GetLength(2);
        var result = new List<double[]>();
        for (int t = 0; t < time; t++)
        {
            var arr = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
                arr[h] = output[0, t, h];
            result.Add(arr);
        }
        return result;
    }

    public void Train(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputSequences.Count; i++)
            {
                var inputSeq = inputSequences[i];
                var targetSeq = targetSequences[i];
                var predicted = PredictSequence(inputSeq);
                // Apply softmax to each prediction
                for (int t = 0; t < predicted.Count; t++)
                    predicted[t] = LossFunctions.Softmax(predicted[t]);

                double seqLoss = 0.0;
                int steps = Math.Min(predicted.Count, targetSeq.Count);
                for (int t = 0; t < steps; t++)
                    seqLoss += LossFunctions.CrossEntropy(predicted[t], targetSeq[t]);
                totalLoss += seqLoss / steps;
                
                Core.Log(Name, Logging.LogLevel.LowDebug, $"seqLoss for sequence {i + 1}/{inputSequences.Count}: {seqLoss / steps}");
                Core.Log(Name, Logging.LogLevel.LowDebug, $"steps: {steps}, predicted: {predicted.Count}, target: {targetSeq.Count}, totalLoss: {totalLoss}");

                // Compute gradient for output
                int time = steps;
                int hiddenSize = predicted[0].Length;
                double[,,] grad = new double[1, time, hiddenSize];
                for (int t = 0; t < time; t++)
                    for (int h = 0; h < hiddenSize; h++)
                        grad[0, t, h] = predicted[t][h] - targetSeq[t][h];

                // Backpropagate through layers in reverse
                for (int l = Layers.Count - 1; l >= 0; l--)
                    grad = Layers[l].Backward(grad, learningRate);
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
            layer.ResetState();
    }

    public double[] Predict(double[] input)
    {
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
