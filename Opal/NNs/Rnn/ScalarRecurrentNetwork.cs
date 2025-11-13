using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Rnn;

public class ScalarRecurrentNetwork : RecurrentNetwork<double, double, double, double, double, double>
{
    public ScalarRecurrentNetwork(
        ActivationFunction<double>? hiddenActivation = null,
        ActivationFunction<double>? outputActivation = null,
        LossFunction<double>? lossFunction = null,
        IOptimizer<double, double>? optimizer = null,
        string name = "ScalarRecurrentNetwork")
        : base(
            [ 1 ],
            [ 1 ],
            [ 1 ],
            1,
            hiddenActivation ?? ActivationFunctions.Identity,
            outputActivation ?? ActivationFunctions.Identity,
            lossFunction ?? LossFunctions.MeanSquaredError,
            optimizer ?? new StandardScalarOptimizer(),
            new ScalarRecurrentTensorOperations(),
            new ScalarRecurrentTensorOperations(),
            new ScalarRecurrentTensorOperations(),
            name)
    {
    }
}

public class ScalarRecurrentTensorOperations : IRecurrentTensorOperations<double, double, double, double, double>
{
    public double DefaultWeights(int[] outputShape, int[] inputShape) => Tensors.RandomDouble();
    public double DefaultBiases(int[] outputShape) => 0.0;
    public double DefaultState(int[] outputsShape) => 0.0;

    public double Add(double a, double b) => a + b;
    public double Add(double a, double b, double c) => a + b + c;
    public double GradInputWeights(double gradZ, double input) => gradZ * input;
    public double GradRecurrentWeights(double gradZ, double state) => gradZ * state;
    public double GradBiases(double gradZ) => gradZ;
    public double GradOutput(double weights, double gradZ) => weights * gradZ;
    public double GradInput(double weights, double gradZ) => weights * gradZ;

    public double Multiply(double a, double b) => a * b;
    public double UpdateState(double output) => output;

    public double ReadWeights(BinaryReader reader, int[] shape) => reader.ReadDouble();
    public void WriteWeights(BinaryWriter writer, double weights) => writer.Write(weights);
    public double ReadBiases(BinaryReader reader, int[] shape) => reader.ReadDouble();
    public void WriteBiases(BinaryWriter writer, double biases) => writer.Write(biases);
    public double ReadState(BinaryReader reader, int[] shape) => reader.ReadDouble();
    public void WriteState(BinaryWriter writer, double state) => writer.Write(state);
}
