using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.NNs.Lstm;

public class VectorLstmNetwork : LstmNetwork<double[,], double[], double[]>
{
    public VectorLstmNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[]> sigmoidActivation, ActivationFunction<double[]> tanhActivation,
        LossFunction<double[][]> lossFunction, IOptimizer<double[,], double[]> optimizer,
        string name = "vector lstm network")
        : base(inputShape, hiddenShape, outputShape, hiddenLayers, sigmoidActivation, tanhActivation, lossFunction,
            optimizer, new StandardVectorTensorOperations(), name)
    {
    }
}