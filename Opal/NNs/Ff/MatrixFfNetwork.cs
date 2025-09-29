using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.NNs.Ff;

public class MatrixFfNetwork : FfNetwork<double[,], double[,], double[,], double[,], double[,]>
{
    public MatrixFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[,]>? hiddenActivation = null,
        ActivationFunction<double[,]>? outputActivation = null,
        LossFunction<double[,]>? lossFunction = null,
        IOptimizer<double[,], double[,]>? optimizer = null,
        string name = "MatrixFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuMatrix,
            outputActivation ?? ActivationFunctions.ReLuMatrix,
            lossFunction ?? LossFunctions.MeanSquaredErrorMatrix,
            optimizer ?? new StandardMatrixOptimizer(),
            new StandardMatrixTensorOperations(),
            new StandardMatrixTensorOperations(),
            new StandardMatrixTensorOperations(),
            name)
    {
    }
}

public class MatrixFfNetworkFactory : 
    IFfNetworkFactory<double[,], double[,], double[,], double[,], double[,], MatrixFfNetwork>
{
    public MatrixFfNetwork Create(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[,]> hiddenActivation, ActivationFunction<double[,]> outputActivation,
        LossFunction<double[,]> lossFunction, IOptimizer<double[,], double[,]> optimizer, string name = "FfNetwork")
    {
        return new MatrixFfNetwork(inputShape, hiddenShape, outputShape, hiddenLayers, hiddenActivation,
            outputActivation, lossFunction, optimizer, name);
    }
}