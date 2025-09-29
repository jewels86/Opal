using Opal.Mathematics;
using Opal.NNs.Ff;

namespace Opal.NNs.Lstm.Attention;


public class LstmAdditiveAttentionLayer<TWeights, TBiases, TTensor> : LstmAttentionLayer<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    private readonly FfNetwork<TWeights, TBiases, TTensor, TTensor, double> alignmentNetwork;
    private readonly List<List<TTensor>> alignmentInputs = new();

    public LstmAdditiveAttentionLayer(
        int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation,
        ActivationFunction<TTensor> tanhActivation,
        FfNetwork<TWeights, TBiases, TTensor, TTensor, double> alignmentNetwork)
        : base(inputShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation)
    {
        this.alignmentNetwork = alignmentNetwork;
    }

    public override double[] Alignment(TTensor[] hidden, TTensor prevState)
    {
        double[] scores = new double[hidden.Length];
        var concatInputs = new List<TTensor>(hidden.Length);
        for (int i = 0; i < hidden.Length; i++)
        {
            TTensor concat = TensorOperations.Concat(hidden[i], prevState);
            concatInputs.Add(concat);
            scores[i] = alignmentNetwork.Forward(concat);
        }
        alignmentInputs.Add(concatInputs);
        return scores;
    }

    public override void TrainAlignment(int timeStep, double[] gradScores, double learningRate)
    {
        var inputs = alignmentInputs.Last();
        alignmentNetwork.Train(inputs.ToArray(), gradScores, 1, learningRate);
    }

    public override TTensor[] Backward(TTensor[] gradOutputs, double learningRate)
    {
        var result = base.Backward(gradOutputs, learningRate);
        alignmentInputs.Clear();
        return result;
    }
}
public class LstmAdditiveAttentionLayerFactory<TWeights, TBiases, TTensor, TNetwork, TNetworkFactory>
    : ILstmAttentionLayerFactory<TWeights, TBiases, TTensor, LstmAdditiveAttentionLayer<TWeights, TBiases, TTensor>>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
    where TNetwork : FfNetwork<TWeights, TBiases, TTensor, TTensor, double>
    where TNetworkFactory : IFfNetworkFactory<TWeights, TBiases, TTensor, TTensor, double, TNetwork>
{
    private readonly TNetworkFactory networkFactory;
    private readonly int[] networkHiddenShape;
    private readonly int networkHiddenLayers;
    private readonly ActivationFunction<TTensor> networkHiddenActivation;
    private readonly ActivationFunction<double> networkOutputActivation;
    private readonly LossFunction<double> networkLossFunction;
    private readonly IOptimizer<TWeights, TBiases> networkOptimizer;

    public LstmAdditiveAttentionLayerFactory(
        TNetworkFactory networkFactory,
        int[] networkHiddenShape,
        int networkHiddenLayers,
        ActivationFunction<TTensor> networkHiddenActivation,
        ActivationFunction<double> networkOutputActivation,
        LossFunction<double> networkLossFunction,
        IOptimizer<TWeights, TBiases> networkOptimizer)
    {
        this.networkFactory = networkFactory;
        this.networkHiddenShape = networkHiddenShape;
        this.networkHiddenLayers = networkHiddenLayers;
        this.networkHiddenActivation = networkHiddenActivation;
        this.networkOutputActivation = networkOutputActivation;
        this.networkLossFunction = networkLossFunction;
        this.networkOptimizer = networkOptimizer;
    }

    public LstmAdditiveAttentionLayer<TWeights, TBiases, TTensor> Create(
        int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation,
        ActivationFunction<TTensor> tanhActivation)
    {
        int[] alignmentInputShape = new int[hiddenShape.Length * 2];
        for (int i = 0; i < hiddenShape.Length; i++)
        {
            alignmentInputShape[i] = hiddenShape[i];
            alignmentInputShape[i + hiddenShape.Length] = hiddenShape[i];
        }

        int[] alignmentOutputShape = [1];

        var alignmentNetwork = networkFactory.Create(
            alignmentInputShape,
            networkHiddenShape,
            alignmentOutputShape,
            networkHiddenLayers,
            networkHiddenActivation,
            networkOutputActivation,
            networkLossFunction,
            networkOptimizer,
            "alignment network");

        return new LstmAdditiveAttentionLayer<TWeights, TBiases, TTensor>(
            inputShape, hiddenShape, outputShape,
            tensorOperations, optimizer,
            sigmoidActivation, tanhActivation,
            alignmentNetwork);
    }
}