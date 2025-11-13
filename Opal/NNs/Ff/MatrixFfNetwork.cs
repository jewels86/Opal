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