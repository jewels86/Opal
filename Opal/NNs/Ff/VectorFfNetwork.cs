using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.NNs.Ff;

public class VectorFfNetwork : FfNetwork<double[,], double[], double[], double[], double[]>
{
    public VectorFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null, 
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null, 
        IOptimizer<double[,], double[]>? optimizer = null,
        string name = "VectorFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector, 
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.CrossEntropy, 
            optimizer ?? new StandardVectorOptimizer(),
            new StandardVectorTensorOperations(),
            new StandardVectorTensorOperations(),
            new StandardVectorTensorOperations(),
            name)
    {
    }
}