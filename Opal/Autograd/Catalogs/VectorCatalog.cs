using System.Numerics;
using Opal.Mathematics;
using Opal.NNs.Ff;
using Opal.NNs.Lstm;
using Opal.NNs.Recurrent;

namespace Opal.Autograd.Catalogs;

public class VectorCatalog : IFfCatalog<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>>
{
    public VectorTensor Multiply(VectorTensor a, MatrixTensor b) => b * a;
    public VectorTensor Multiply(VectorTensor a, VectorTensor b) => a * b;
    public VectorTensor Add(VectorTensor a, VectorTensor b) => a + b;
    public VectorTensor Subtract(VectorTensor a, VectorTensor b) => a - b;
    
    public ITensorStorage<double[]> Scale(ITensorStorage<double[]> a, ITensorStorage<double> scale) => Operations.MultiplyScalarStorage(a, scale);

    public ITensorStorage<double[]> ZeroGradient(ITensorStorage<double[]> a) => new CpuStorage<double[]>(Vectors.Zeros(a.TotalElements), [a.TotalElements], a.TotalElements);

    public VectorTensor ConcatHidden(VectorTensor input, VectorTensor prevHidden) => Operations.Concat(input, prevHidden);

    public VectorTensor ConcatInputHidden(VectorTensor input, VectorTensor prevHidden) => Operations.Concat(input, prevHidden);

    public VectorTensor DefaultHidden(int size) => new(
        new CpuStorage<double[]>(Vectors.Zeros(size), [size], size), 
        null, _ => { }, new CpuStorage<double[]>(Vectors.Zeros(size), [size], size)
        );

    public VectorTensor DefaultState(int size) => DefaultHidden(size);

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
    
    public void WriteState(BinaryWriter writer, double[] state)
    {
        writer.Write(state.Length);
        foreach (var s in state)
            writer.Write(s);
    }

    public double[] ReadState(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var state = new double[length];
        for (int i = 0; i < length; i++)
            state[i] = reader.ReadDouble();
        return state;
    }
}

