namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmAdditiveAttentionLayer : LstmAttentionLayer<LstmAdditiveAttentionLayer.LstmAdditiveAttentionBackpropCache>
{
    private FfLayer alignmentLayer;
    private double[] v;

    public LstmAdditiveAttentionLayer(int inputSize, int hiddenSize, int attentionSize, int alignmentHiddenSize = 32, string name = "LstmAdditiveAttentionLayer")
        : base(inputSize, hiddenSize, attentionSize, name)
    {
        alignmentLayer = new FfLayer(hiddenSize * 2, alignmentHiddenSize, MathFunctions.Tanh, MathFunctions.TanhDerivative);
        v = new double[alignmentHiddenSize];
        var rand = new Random();
        for (int i = 0; i < v.Length; i++)
            v[i] = rand.NextDouble() * 2 - 1;
    }

    public override double[] Alignment(double[] encoderHidden, double[] decoderHidden, Action<object>? alignmentCacheAction = null)
    {
        double[] concat = encoderHidden.Concat(decoderHidden).ToArray();
        double[] ff = alignmentLayer.Forward(concat);
        double score = 0.0;
        for (int i = 0; i < v.Length; i++)
            score += v[i] * ff[i];
        alignmentCacheAction?.Invoke((ff, concat));
        return [score];
    }

    public override (Dictionary<string, object>, Action<object>) PrepareToCacheAlignment()
    {
        var cache = new Dictionary<string, object>();
        List<double[]> ffList = [];
        List<double[]> concatList = [];
        cache["ffList"] = ffList;
        cache["concatList"] = concatList;
        return (cache, obj => {
            if (obj is ValueTuple<double[], double[]> tuple)
            {
                ffList.Add(tuple.Item1);
                concatList.Add(tuple.Item2);
            }
        });
    }

    public override void FinalizeAlignmentCache(Dictionary<string, object> alignmentCache)
    {
        BackpropCache.FfList = alignmentCache["ffList"] as List<double[]> ?? [];
        BackpropCache.ConcatList = alignmentCache["concatList"] as List<double[]> ?? [];
    }

    public override void TrainAlignment(LstmAttentionBackpropCache cache, int decoderTimeStep, double[] gradScores, double learningRate)
    {
        if (cache is LstmAdditiveAttentionBackpropCache addCache)
        {
            var ffList = addCache.FfList;
            var concatList = addCache.ConcatList;
            if (decoderTimeStep < ffList.Count && decoderTimeStep < concatList.Count)
            {
                double[] ff = ffList[decoderTimeStep];
                for (int i = 0; i < v.Length; i++)
                    v[i] -= learningRate * gradScores[0] * ff[i];
                double[] gradFf = new double[v.Length];
                for (int i = 0; i < v.Length; i++)
                    gradFf[i] = gradScores[0] * v[i];
                alignmentLayer.Backward(gradFf, learningRate);
            }
        }
    }

    public override void SaveAlignment(BinaryWriter writer)
    {
        for (int i = 0; i < v.Length; i++)
            writer.Write(v[i]);
        for (int i = 0; i < alignmentLayer.Weights.GetLength(0); i++)
            for (int j = 0; j < alignmentLayer.Weights.GetLength(1); j++)
                writer.Write(alignmentLayer.Weights[i, j]);
        for (int i = 0; i < alignmentLayer.Biases.Length; i++)
            writer.Write(alignmentLayer.Biases[i]);
    }

    public override void LoadAlignment(BinaryReader reader)
    {
        for (int i = 0; i < v.Length; i++)
            v[i] = reader.ReadDouble();
        for (int i = 0; i < alignmentLayer.Weights.GetLength(0); i++)
            for (int j = 0; j < alignmentLayer.Weights.GetLength(1); j++)
                alignmentLayer.Weights[i, j] = reader.ReadDouble();
        for (int i = 0; i < alignmentLayer.Biases.Length; i++)
            alignmentLayer.Biases[i] = reader.ReadDouble();
    }

    public override void ResetAlignment()
    {
        alignmentLayer.Reset();
        var rand = new Random();
        for (int i = 0; i < v.Length; i++)
            v[i] = rand.NextDouble() * 2 - 1;
    }

    public class LstmAdditiveAttentionBackpropCache : LstmAttentionBackpropCache
    {
        public List<double[]> FfList { get; set; } = [];
        public List<double[]> ConcatList { get; set; } = [];
    }
}