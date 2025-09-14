namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmDotAttentionLayer : LstmAttentionLayer<LstmDotAttentionLayer.LstmDotAttentionBackpropCache>
{
    public LstmDotAttentionLayer(int inputSize, int hiddenSize, int attentionSize, string name = "LstmDotAttentionLayer") 
        : base(inputSize, hiddenSize, attentionSize, name)
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

    public override void FinalizeAttentionCache(Dictionary<string, object> alignmentCache)
    {
        
    }

    public override void TrainAlignment(LstmAttentionBackpropCache cache, int decoderTimeStep, double[] gradScores, double learningRate)
    {
        
    }

    public override void LoadAttention(BinaryReader reader)
    {
        
    }
    public override void SaveAttention(BinaryWriter writer)
    {
        
    }

    public override void ResetAttention()
    {
        
    }
    #endregion

    public class LstmDotAttentionBackpropCache : LstmAttentionBackpropCache
    {
        
    }
}