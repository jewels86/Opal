using Opal.Mathematics;
using Opal.Utilities.ANNs.Lstm;

namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmDotAttentionLayer<TWeights, TBiases, TTensor> : LstmAttentionLayer<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    public LstmDotAttentionLayer(
        int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation,
        ActivationFunction<TTensor> tanhActivation)
        : base(inputShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation)
    {
    }

    public override double[] Alignment(TTensor[] hidden, TTensor prevState)
    {
        double[] scores = new double[hidden.Length];
        for (int i = 0; i < hidden.Length; i++)
        {
            scores[i] = TensorOperations.Dot(hidden[i], prevState);
        }
        return scores;
    }

    public override void TrainAlignment(int timeStep, double[] gradScores, double learningRate) { }
}

public class LstmDotAttentionLayerFactory<TWeights, TBiases, TTensor> : ILstmAttentionLayerFactory<TWeights, TBiases, TTensor, LstmDotAttentionLayer<TWeights, TBiases, TTensor>>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    public LstmDotAttentionLayer<TWeights, TBiases, TTensor> Create(
        int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation,
        ActivationFunction<TTensor> tanhActivation)
    {
        return new LstmDotAttentionLayer<TWeights, TBiases, TTensor>(
            inputShape, hiddenShape, outputShape,
            tensorOperations, optimizer,
            sigmoidActivation, tanhActivation);
    }
}