namespace Opal.Utilities.ANNs.Lstm;

public class LstmAttentionNetwork<T> where T : LstmAttentionBackpropCache, new()
{
    public string Name { get; set; }
    public List<LstmAttentionLayer<T>> Layers { get; set; }

    public LstmAttentionNetwork(string name)
    {
        Name = name;
        Layers = [];
    }

    public void AddLayer(LstmAttentionLayer<T> layer) => Layers.Add(layer);

    public List<double[]> PredictSequence(List<double[]> inputSequence, List<double[]> outputSequence)
    {
        if (inputSequence.Count == 0 || outputSequence.Count == 0)
            return [];

        int time = inputSequence.Count;
        int inputSize = inputSequence[0].Length;

        double[,,] input = new double[1, time, inputSize];
        for (int t = 0; t < time; t++)
        for (int i = 0; i < inputSize; i++)
            input[0, t, i] = inputSequence[t][i];
        
        double[,] output = MathFunctions.GetBatchSample(input, 0);
        foreach (var layer in Layers)
            output = layer.Forward(output, MathFunctions.ToMatrix2D(outputSequence));

        return MathFunctions.ToVectorList(output);
    }

    public List<double> Train(List<List<double[]>> inputSequences, List<List<double[]>> outputSequences, List<List<double[]>> targetSequences, int epochs, double learningRate)
    {
        var epochLosses = new List<double>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            for (int i = 0; i < inputSequences.Count; i++)
            {
                var inputSeq = inputSequences[i];
                var outputSeq = outputSequences[i];
                var targetSeq = targetSequences[i];
                var predicted = PredictSequence(inputSeq, outputSeq).Select(x => LossFunctions.Softmax(x)).ToList();
                var reversedPredicted = ((IEnumerable<double[]>)predicted).Reverse().ToList();
                var reversedTarget = ((IEnumerable<double[]>)targetSeq).Reverse().ToList();
                double seqLoss = 0.0;
                int steps = Math.Min(reversedPredicted.Count, reversedTarget.Count);
                for (int t = 0; t < steps; t++)
                    seqLoss += LossFunctions.CrossEntropy(reversedPredicted[t], reversedTarget[t]);
                totalLoss += seqLoss / steps;
                int time = steps;
                int outputSize = reversedPredicted[0].Length;
                double[,] grad = new double[time, outputSize];
                for (int t = 0; t < time; t++)
                {
                    for (int h = 0; h < outputSize; h++)
                        grad[t, h] = reversedPredicted[t][h] - reversedTarget[t][h];
                }
                for (int l = Layers.Count - 1; l >= 0; l--)
                    Layers[l].Backward(grad, learningRate);
            }
            double averageLoss = totalLoss / inputSequences.Count;
            epochLosses.Add(averageLoss);
            // Optionally log here
        }
        return epochLosses;
    }

    public void Reset()
    {
        foreach (var layer in Layers)
        {
            layer.Reset();
        }
    }

    public void Save(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(fs);
        writer.Write(Name);
        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            layer.Save(writer);
        }
    }
}
