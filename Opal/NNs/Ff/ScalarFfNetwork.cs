using Opal.Mathematics;
using Opal.Mathematics.TensorOperations;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : FfNetwork<double, double, double, double, double>
{
    public ScalarFfNetwork(
        ActivationFunction<double>? hiddenActivation = null,
        ActivationFunction<double>? outputActivation = null,
        LossFunction<double>? lossFunction = null,
        IOptimizer<double, double>? optimizer = null,
        string name = "ScalarFfNetwork")
        : base(
            [1],
            [1],
            [1],
            1,
            hiddenActivation ?? ActivationFunctions.Identity,
            outputActivation ?? ActivationFunctions.Identity,
            lossFunction ?? LossFunctions.MeanSquaredError,
            optimizer ?? new StandardScalarOptimizer(),
            new StandardScalarTensorOperations(),
            new StandardScalarTensorOperations(),
            new StandardScalarTensorOperations(),
            name)
    {
    }
}