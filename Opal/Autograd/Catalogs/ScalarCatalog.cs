using System.Numerics;
using Opal.Mathematics;
using Opal.NNs.Ff;
using Opal.Utilities;

namespace Opal.Autograd.Catalogs;

public class ScalarCatalog 
{
    public ScalarTensor Multiply(VectorTensor a, ScalarTensor b) => Operations.Sum(Operations.Multiply(a, b));
    public ScalarTensor Add(ScalarTensor a, ScalarTensor b) => Operations.Add(a, b);
    public VectorTensorStorage Subtract(VectorTensorStorage a, VectorTensorStorage b) => Operations.SubtractStorage(a, b);
    public ScalarTensorStorage Subtract(ScalarTensorStorage a, ScalarTensorStorage b) => Operations.SubtractStorage(a, b);
    public ScalarTensorStorage Scale(ScalarTensorStorage a, double scale) => Operations.MultiplyStorage(a, Operations.NewDefaultScalarStorage(scale));
    public VectorTensorStorage Scale(VectorTensorStorage a, double scale) => Operations.ScaleVectorStorage(a, Operations.NewDefaultScalarStorage(scale));
    public ScalarTensorStorage ZeroGradient(ScalarTensorStorage a) => Operations.NewDefaultScalarStorage(0.0);
    public VectorTensorStorage ZeroGradient(VectorTensorStorage a) => Operations.NewDefaultVectorStorage(Vectors.Zeros(a.TotalElements));
    public ScalarTensorStorage ReadBias(BinaryReader reader) => Operations.NewDefaultScalarStorage(reader.ReadDouble());
    public VectorTensorStorage ReadWeights(BinaryReader reader) => Operations.NewDefaultVectorStorage(BinaryWriting.ReadVector(reader));
    public void WriteBias(BinaryWriter writer, ScalarTensorStorage bias) => writer.Write(bias.ToHost());
    public void WriteWeights(BinaryWriter writer, VectorTensorStorage weights) => BinaryWriting.WriteVector(writer, weights.ToHost());
}