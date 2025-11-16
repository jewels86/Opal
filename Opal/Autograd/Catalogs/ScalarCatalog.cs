using System.Numerics;
using Opal.Mathematics;
using Opal.NNs.Ff;
using Opal.NNs.Recurrent;
using Opal.Utilities;

namespace Opal.Autograd.Catalogs;

public class ScalarCatalog : IFfCatalog<ITensorStorage<double>, ITensorStorage<double>, VectorTensorStorage>
{
    public ScalarTensor Multiply(VectorTensor a, ScalarTensor b) => Operations.Sum(Operations.Multiply(a, b));
    public ScalarTensor Add(ScalarTensor a, ScalarTensor b) => Operations.Add(a, b);
    public VectorTensorStorage Subtract(VectorTensorStorage a, VectorTensorStorage b) => Operations.SubtractStorage(a, b);
    public ITensorStorage<double> Subtract(ITensorStorage<double> a, ITensorStorage<double> b) => Operations.SubtractStorage(a, b);
    public ITensorStorage<double> Scale(ITensorStorage<double> a, double scale) => Operations.MultiplyStorage(a, Operations.NewDefaultScalarStorage(scale));
    public VectorTensorStorage Scale(VectorTensorStorage a, double scale) => Operations.ScaleVectorStorage(a, Operations.NewDefaultScalarStorage(scale));
    public ITensorStorage<double> ZeroGradient(ITensorStorage<double> a) => Operations.NewDefaultScalarStorage(0.0);
    public VectorTensorStorage ZeroGradient(VectorTensorStorage a) => Operations.NewDefaultVectorStorage(Vectors.Zeros(a.TotalElements));
    public ITensorStorage<double> ReadBias(BinaryReader reader) => Operations.NewDefaultScalarStorage(reader.ReadDouble());
    public VectorTensorStorage ReadWeights(BinaryReader reader) => Operations.NewDefaultVectorStorage(BinaryWriting.ReadVector(reader));
    public void WriteBias(BinaryWriter writer, ITensorStorage<double> bias) => writer.Write(bias.ToHost());
    public void WriteWeights(BinaryWriter writer, VectorTensorStorage weights) => BinaryWriting.WriteVector(writer, weights.ToHost());
}