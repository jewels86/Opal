using System.IO;

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

    public List<double> Train(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences, int epochs, double learningRate)
    {
        var epochLosses = new List<double>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputSequences.Count; i++)
            {
                var inputSeq = inputSequences[i];
                var targetSeq = targetSequences[i];
                if (inputSeq.Count != targetSeq.Count)
                {
                    Core.Log(Name, 2, "Input and target sequences must have the same length.");
                    continue;
                }
                
                var predicted = PredictSequence(inputSeq).Select(x => LossFunctions.Softmax(x)).ToList();

                // Reverse, then iterate through so that for nwg last is the only one considered and for transformations it works anyway
                var reversedPredicted = ((IEnumerable<double[]>)predicted).Reverse().ToList();
                var reversedTarget = ((IEnumerable<double[]>)targetSeq).Reverse().ToList();

                double seqLoss = 0.0;
                int steps = Math.Min(reversedPredicted.Count, reversedTarget.Count);
                for (int t = 0; t < steps; t++) 
                    seqLoss += LossFunctions.CrossEntropy(reversedPredicted[t], reversedTarget[t]);
                totalLoss += seqLoss / steps;

                int time = steps;
                int hiddenSize = reversedPredicted[0].Length;
                double[,,] grad = new double[1, time, hiddenSize];
                for (int t = 0; t < time; t++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                        grad[0, t, h] = reversedPredicted[t][h] - reversedTarget[t][h];
                }
                for (int l = Layers.Count - 1; l >= 0; l--)
                    grad = Layers[l].Backward(grad, learningRate);
            }
            double averageLoss = totalLoss / inputSequences.Count;
            epochLosses.Add(averageLoss);
            Core.Log(Name, Logging.LogLevel.HighDebug, $"Epoch {epoch + 1}/{epochs}, Loss: {averageLoss}");
        }
        return epochLosses;
    }

    public double EvaluateLoss(List<List<double[]>> inputSequences, List<List<double[]>> targetSequences)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputSequences.Count; i++)
        {
            var predicted = PredictSequence(inputSequences[i]);
            for (int t = 0; t < predicted.Count; t++) 
                predicted[t] = LossFunctions.Softmax(predicted[t]);
            
            var actual = targetSequences[i];
            double seqLoss = LossFunctions.CrossEntropy(predicted.Last(), actual[0]);
            totalLoss += seqLoss;
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
        var outputSeq = PredictSequence([input]);
        return outputSeq.Count > 0 ? outputSeq[0] : [];
    }

    public List<double> Train(double[][] inputs, double[][] targets, int epochs, double learningRate)
    {
        var inputSeqs = inputs.Select(x => new List<double[]> { x }).ToList();
        var targetSeqs = targets.Select(x => new List<double[]> { x }).ToList();
        return Train(inputSeqs, targetSeqs, epochs, learningRate);
    }

    public double EvaluateLoss(double[][] inputs, double[][] targets)
    {
        var inputSeqs = inputs.Select(x => new List<double[]> { x }).ToList();
        var targetSeqs = targets.Select(x => new List<double[]> { x }).ToList();
        return EvaluateLoss(inputSeqs, targetSeqs);
    }

    public void Save(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(fs);
        writer.Write(Name ?? "");
        writer.Write(Layers.Count);
        foreach (var layer in Layers)
            layer.Save(writer);
    }

    public static LstmNetwork Load(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        string name = reader.ReadString();
        int layerCount = reader.ReadInt32();
        var net = new LstmNetwork(name);
        for (int i = 0; i < layerCount; i++)
            net.Layers.Add(LstmLayer.Load(reader));
        return net;
    }
}
