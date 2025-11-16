using System.Numerics;
using Opal.Mathematics;
using Opal.NNs.Ff;
using Opal.NNs.Lstm;
using Opal.NNs.Recurrent;
using Opal.Utilities;

namespace Opal.Autograd.Catalogs;

public class VectorCatalog : IFfCatalog<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>>
{
    public VectorTensor Add(VectorTensor a, VectorTensor b) => Operations.Add(a, b);
    public VectorTensor Multiply(MatrixTensor a, VectorTensor b) => Operations.Multiply(a, b);
    public ITensorStorage<double[,]> Subtract(ITensorStorage<double[,]> a, ITensorStorage<double[,]> b) => Operations.SubtractStorage(a, b);
    public ITensorStorage<double[]> Subtract(ITensorStorage<double[]> a, ITensorStorage<double[]> b) => Operations.SubtractStorage(a, b);
    public ITensorStorage<double[,]> Scale(ITensorStorage<double[,]> a, double scale)
    {
        return Operations.ScaleMatrixStorage(a, Operations.NewDefaultScalarStorage(scale));
    }

    public ITensorStorage<double[]> Scale(ITensorStorage<double[]> a, double scale) => Operations.ScaleVectorStorage(a, Operations.NewDefaultScalarStorage(scale));

    public ITensorStorage<double[]> ZeroGradient(ITensorStorage<double[]> a) => Operations.NewDefaultVectorStorage(Vectors.Zeros(a.TotalElements));
    public ITensorStorage<double[,]> ZeroGradient(ITensorStorage<double[,]> a) => Operations.NewDefaultMatrixStorage(Matrices.Zeros(a.Shape[0], a.Shape[1]));

    public ITensorStorage<double[]> ReadBias(BinaryReader reader) => Operations.NewDefaultVectorStorage(BinaryWriting.ReadVector(reader));
    public ITensorStorage<double[,]> ReadWeights(BinaryReader reader) => Operations.NewDefaultMatrixStorage(BinaryWriting.ReadMatrix(reader));
    public void WriteBias(BinaryWriter writer, ITensorStorage<double[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, ITensorStorage<double[,]> weights) => BinaryWriting.WriteMatrix(writer, weights.ToHost());
}

