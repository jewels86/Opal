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

public class VectorFfNetworkFactory : 
    IFfNetworkFactory<double[,], double[], double[], double[], double[], VectorFfNetwork>
{
    public VectorFfNetwork Create(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<double[]> hiddenActivation, ActivationFunction<double[]> outputActivation,
        LossFunction<double[]> lossFunction, IOptimizer<double[,], double[]> optimizer, string name = "FfNetwork")
    {
        return new VectorFfNetwork(inputShape, hiddenShape, outputShape, hiddenLayers, hiddenActivation,
            outputActivation, lossFunction, optimizer, name);
    }
}