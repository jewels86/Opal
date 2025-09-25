using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class ScalarFfNetwork : FfNetwork<double, double, double, double, double>
{
    public ScalarFfNetwork(
        ActivationFunction<double>? hiddenActivation = null,
        ActivationFunction<double>? outputActivation = null,
        LossFunction<double>? lossFunction = null,
        IOptimizer<double, double>? optimizer = null,
        string name = "ScalarFfNetwork")
        : base(
            new int[] { 1 },
            new int[] { 1 },
            new int[] { 1 },
            1,
            hiddenActivation ?? ActivationFunctions.Identity,
            outputActivation ?? ActivationFunctions.Identity,
            lossFunction ?? LossFunctions.MeanSquaredError,
            optimizer ?? new StandardScalarOptimizer(),
            new ScalarFfTensorOperations(),
            new ScalarFfTensorOperations(),
            new ScalarFfTensorOperations(),
            name)
    {
    }
}

public class ScalarFfTensorOperations : IFfTensorOperations<double, double, double, double>
{
    public double Add(double output, double biases) => output + biases;
    public double Apply(double output, Func<double, double> activation) => activation(output);
    public double DefaultBiases(int[] shape) => 0.0;
    public double DefaultOutput(int[] shape) => 0.0;
    public double DefaultWeights(int[] outputShape, int[] inputShape) => Tensors.RandomDouble();
    public double DefaultInput(int[] shape) => 0.0;
    public double Multiply(double weights, double input) => weights * input;
    public double GradBiases(double gradZ) => gradZ;
    public double GradInput(double weights, double gradZ) => weights * gradZ;
    public double GradWeights(double gradZ, double lastInput) => gradZ * lastInput;
}
