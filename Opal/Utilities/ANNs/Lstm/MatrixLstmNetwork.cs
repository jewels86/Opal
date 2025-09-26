using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.Utilities.ANNs.Lstm;

public class MatrixLstmNetwork : LstmNetwork<double[,], double[,], double[,]>
{
    public MatrixLstmNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[,]> sigmoidActivation, ActivationFunction<double[,]> tanhActivation,
        LossFunction<double[][,]> lossFunction, IOptimizer<double[,], double[,]> optimizer,
        string name = "vector lstm network")
        : base(inputShape, hiddenShape, outputShape, hiddenLayers, sigmoidActivation, tanhActivation, lossFunction,
            optimizer, new StandardMatrixTensorOperations(), name)
    {
    }
}