using Opal.NNs.Ff;

namespace Opal.Autograd.Catalogs;

public class ScalarCatalog : IFfCatalog<double, double, double>
{
    public Tensor<double> Multiply(Tensor<double> input, Tensor<double>[] weights)
    {
        var products = weights.Select(w => Operations.Multiply(input, w)).ToList();
        return Operations.Sum(products);
    }

    public Tensor<double> Add(Tensor<double> a, Tensor<double> b) => Operations.Sum(a, b);

    public double Subtract(double a, double b) => a - b;
    
    public double Scale(double a, double scale) => a * scale;
    
    public double ZeroGradient(double a) => 0.0;

    public void WriteWeight(BinaryWriter writer, double weight) => writer.Write(weight);

    public double ReadWeight(BinaryReader reader) => reader.ReadDouble();

    public void WriteBias(BinaryWriter writer, double bias) => writer.Write(bias);

    public double ReadBias(BinaryReader reader) => reader.ReadDouble();
}