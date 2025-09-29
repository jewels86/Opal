using Opal.Mathematics;

namespace Opal.NNs.Lstm.Attention;

public class LstmAttentionNetwork<TWeights, TBiases, TTensor, TLayer, TLayerFactory> : LstmNetwork<TWeights, TBiases, TTensor>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
    where TLayer : LstmAttentionLayer<TWeights, TBiases, TTensor>
    where TLayerFactory : ILstmAttentionLayerFactory<TWeights, TBiases, TTensor, TLayer>
{
    public new TLayer InputLayer { get; }
    public new List<TLayer> HiddenLayers { get; }
    public new TLayer OutputLayer { get; }
    
    public LstmAttentionNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation,
        LossFunction<TTensor[]> lossFunction, IOptimizer<TWeights, TBiases> optimizer,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations, TLayerFactory layerFactory,
        string name = "lstm attention network")
        : base(inputShape, hiddenShape, outputShape, hiddenLayers, sigmoidActivation, tanhActivation, lossFunction,
            optimizer, tensorOperations, name)
    {
        InputLayer = layerFactory.Create(inputShape, hiddenShape, hiddenShape, tensorOperations, optimizer, sigmoidActivation,
            tanhActivation);
        HiddenLayers = [];
        for (int i = 0; i < hiddenLayers; i++)
            HiddenLayers.Add(layerFactory.Create(hiddenShape, hiddenShape, hiddenShape, tensorOperations, optimizer, sigmoidActivation,
                tanhActivation));
        OutputLayer = layerFactory.Create(hiddenShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation,
            tanhActivation);
    }
    
    public override TTensor[] Forward(TTensor[] input)
    {
        var output = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            output = layer.Forward(output);
        return OutputLayer.Forward(output);
    }
    
    public override void Train(TTensor[][] inputs, TTensor[][] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var target = targets[i];
                var hidden = InputLayer.Forward(input);
                foreach (var layer in HiddenLayers)
                    hidden = layer.Forward(hidden);
                var output = OutputLayer.Forward(hidden);

                var lossGrad = LossFunction.Derivative(output, target);

                var grad = OutputLayer.Backward(lossGrad, learningRate);
                for (int h = HiddenLayers.Count - 1; h >= 0; h--)
                    grad = HiddenLayers[h].Backward(grad, learningRate);
                InputLayer.Backward(grad, learningRate);
            }
        }
    }
}

public interface ILstmAttentionLayerFactory<TWeights, TBiases, TTensor, TLayer>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
    where TLayer : LstmAttentionLayer<TWeights, TBiases, TTensor>
{
    public TLayer Create(int[] inputShape, int[] hiddenShape, int[] outputShape,
        ILstmAttentionTensorOperations<TWeights, TBiases, TTensor> tensorOperations, IOptimizer<TWeights, TBiases> optimizer,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation);
}