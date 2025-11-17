using System.Numerics;
using Opal.Mathematics;
using Opal.NNs.Ff;
using Opal.Utilities;

namespace Opal.Autograd.Catalogs;

public class VectorCatalog : IFfCatalog<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>
{
    public VectorTensor Add(VectorTensor a, VectorTensor b) => Operations.Add(a, b);
    public VectorTensor Multiply(MatrixTensor a, VectorTensor b) => Operations.Multiply(a, b);
    public MatrixTensorStorage Subtract(MatrixTensorStorage a, MatrixTensorStorage b) => Operations.SubtractStorage(a, b);
    public VectorTensorStorage Subtract(VectorTensorStorage a, VectorTensorStorage b) => Operations.SubtractStorage(a, b);
    public MatrixTensorStorage Scale(MatrixTensorStorage a, double scale)
    {
        return Operations.ScaleMatrixStorage(a, Operations.NewDefaultScalarStorage(scale));
    }

    public VectorTensorStorage Scale(VectorTensorStorage a, double scale) => Operations.ScaleVectorStorage(a, Operations.NewDefaultScalarStorage(scale));

    public VectorTensorStorage ZeroGradient(VectorTensorStorage a) => Operations.NewDefaultVectorStorage(Vectors.Zeros(a.TotalElements));
    public MatrixTensorStorage ZeroGradient(MatrixTensorStorage a) => Operations.NewDefaultMatrixStorage(Matrices.Zeros(a.Shape[0], a.Shape[1]));

    public VectorTensorStorage ReadBias(BinaryReader reader) => Operations.NewDefaultVectorStorage(BinaryWriting.ReadVector(reader));
    public MatrixTensorStorage ReadWeights(BinaryReader reader) => Operations.NewDefaultMatrixStorage(BinaryWriting.ReadMatrix(reader));
    public void WriteBias(BinaryWriter writer, VectorTensorStorage bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, MatrixTensorStorage weights) => BinaryWriting.WriteMatrix(writer, weights.ToHost());
}

