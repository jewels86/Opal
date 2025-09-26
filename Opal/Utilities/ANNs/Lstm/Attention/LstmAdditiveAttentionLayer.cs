using Opal.Mathematics;
using Opal.Utilities.ANNs.Ff;
using Opal.Utilities.ANNs.Lstm;

namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmAdditiveAttentionLayer<TWeights, TBiases, TTensor> : LstmAttentionLayer<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    private readonly FfNetwork<TWeights, TBiases, TTensor, TTensor, double[]> alignmentNetwork;
    private readonly List<TTensor[]> alignmentInputs = [];
    
    public LstmAdditiveAttentionLayer(
        int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation,
        ActivationFunction<TTensor> tanhActivation,
        FfNetwork<TWeights, TBiases, TTensor, TTensor, double[]> alignmentNetwork)
        : base(inputShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation)
    {
        this.alignmentNetwork = alignmentNetwork;
    }

    public override double[] Alignment(TTensor[] hidden, TTensor prevState)
    {
        double[] scores = new double[hidden.Length];
        TTensor[] concatInputs = new TTensor[hidden.Length];
        for (int i = 0; i < hidden.Length; i++)
        {
            TTensor concat = TensorOperations.Concat(hidden[i], prevState);
            concatInputs[i] = concat;
            scores[i] = alignmentNetwork.Forward(concat)[0];
        }
        alignmentInputs.Add(concatInputs);
        return scores;
    }

    public override void TrainAlignment(int timeStep, double[] gradScores, double learningRate)
    {
        var inputs = alignmentInputs[timeStep];
        var targets = gradScores.Select(g => new[] { g }).ToArray();
        alignmentNetwork.Train(inputs, targets, 1, learningRate);
        
    }
    
    public override TTensor[] Backward(TTensor[] gradOutputs, double learningRate)
    {
        var result = base.Backward(gradOutputs, learningRate);
        alignmentInputs.Clear();
        return result;
    }
}