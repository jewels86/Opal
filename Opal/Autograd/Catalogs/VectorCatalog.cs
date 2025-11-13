using Opal.Mathematics;
using Opal.NNs.Ff;

namespace Opal.Autograd.Catalogs;

public class VectorCatalog : IFfCatalog<double[], double[], double[]>
{
    public Tensor<double[]> Multiply(Tensor<double[]> a, Tensor<double[]>[] b) => Operations.Multiply(a, b);

    public Tensor<double[]> Add(Tensor<double[]> a, Tensor<double[]> b) => 
        Operations.Sum(a, b);

    public double[] Subtract(double[] a, double[] b) => Vectors.Subtract(a, b);
    
    public double[] Scale(double[] a, double scale) => Vectors.Multiply(a, scale);
    
    public double[] ZeroGradient(double[] a) => Vectors.Zeros(a.Length);

    public void WriteWeight(BinaryWriter writer, double[] weight)
    {
        writer.Write(weight.Length);
        foreach (var w in weight)
            writer.Write(w);
    }

    public double[] ReadWeight(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var weight = new double[length];
        for (int i = 0; i < length; i++)
            weight[i] = reader.ReadDouble();
        return weight;
    }

    public void WriteBias(BinaryWriter writer, double[] bias)
    {
        writer.Write(bias.Length);
        foreach (var b in bias)
            writer.Write(b);
    }

    public double[] ReadBias(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var bias = new double[length];
        for (int i = 0; i < length; i++)
            bias[i] = reader.ReadDouble();
        return bias;
    }
}

