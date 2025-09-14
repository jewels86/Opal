namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmDotAttentionLayer : LstmAttentionLayer<LstmDotAttentionLayer.LstmDotAttentionBackpropCache>
{
    public LstmDotAttentionLayer(int inputSize, int hiddenSize, int outputSize, string name = "LstmDotAttentionLayer") 
        : base(inputSize, hiddenSize, outputSize, name)
    {
    }

    public override double[] Alignment(double[] encoderHidden, double[] decoderHidden, Action<object>? alignmentCacheAction = null)
    {
        double score = 0.0;
        for (int i = 0; i < encoderHidden.Length; i++)
            score += encoderHidden[i] * decoderHidden[i];
        return [score];
    }

    #region Overrides
    public override (Dictionary<string, object>, Action<object>) PrepareToCacheAlignment()
    {
        return ([], x => { });
    }

    public override void FinalizeAlignmentCache(Dictionary<string, object> alignmentCache)
    {
        
    }

    public override void TrainAlignment(LstmAttentionBackpropCache cache, int decoderTimeStep, double[] gradScores, double learningRate)
    {
        
    }

    public override void LoadAlignment(BinaryReader reader)
    {
        
    }
    public override void SaveAlignment(BinaryWriter writer)
    {
        
    }

    public override void ResetAlignment()
    {
        
    }
    #endregion

    public class LstmDotAttentionBackpropCache : LstmAttentionBackpropCache
    {
        
    }
}

public class LstmDotAttentionNetwork : LstmAttentionNetwork<LstmDotAttentionLayer.LstmDotAttentionBackpropCache>
{
    public LstmDotAttentionNetwork(string name = "LstmDotAttentionNetwork") : base(name)
    {
    }
}