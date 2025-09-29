using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.Utilities.ANNs.Lstm.Attention;

public class LstmDotVectorAttentionNetwork : 
    LstmAttentionNetwork<double[,], double[], double[], 
        LstmDotAttentionLayer<double[,], double[], double[]>, 
        LstmDotAttentionLayerFactory<double[,], double[], double[]>>
{
    public LstmDotVectorAttentionNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[]> sigmoidActivation, ActivationFunction<double[]> tanhActivation,
        LossFunction<double[][]> lossFunction, IOptimizer<double[,], double[]> optimizer,
        string name = "lstm dot vector attention network")
        : base(inputShape, hiddenShape, outputShape, hiddenLayers, sigmoidActivation, tanhActivation, lossFunction,
            optimizer, new StandardVectorTensorOperations(), new LstmDotAttentionLayerFactory<double[,], double[], double[]>(), name)
    {
    }
}