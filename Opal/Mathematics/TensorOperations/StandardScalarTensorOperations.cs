using Opal.NNs.Ff;

namespace Opal.Mathematics.TensorOperations;

public class StandardScalarTensorOperations : IFfTensorOperations<double, double, double, double>
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

    public double ReadBiases(BinaryReader reader, int[] shape) => reader.ReadDouble();
    public void WriteBiases(BinaryWriter writer, double biases) => writer.Write(biases);
    public double ReadWeights(BinaryReader reader, int[] shape) => reader.ReadDouble();
    public void WriteWeights(BinaryWriter writer, double weights) => writer.Write(weights);
}
