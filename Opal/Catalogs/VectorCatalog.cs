using System.Diagnostics;
using Jewels.Lazulite;
using Opal.NNs.Ff;
using Opal.Utilities;

namespace Opal;

public class VectorCatalog : IFfCatalog<float[], float[], float[,]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;

    public Tensor<float[]> Multiply(Tensor<float[,]> a, Tensor<float[]> b)
    {
        var aidx = a.AcceleratorIndex;
        var result = new VectorValue(Compute.Get(aidx, a.Value.Shape[0] * b.Value.Shape[1]));
        Compute.MatrixMultiply(a.Value, b.Value, result, a.Value.Shape[0], b.Value.Shape[1], b.Value.Shape[0]);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var gradA = Compute.Get(aidx, a.Value.TotalSize);
            Compute.MatrixMultiply(tensor.Gradient.Data, b.Value.Data, gradA, 
                a.Value.Shape[0], b.Value.Shape[0], b.Value.Shape[1], transposeB: true);
            Compute.Call(aidx, Compute.ElementwiseAddKernels, a.Gradient.Data, gradA, a.Gradient.Data);

            var gradB = Compute.Get(aidx, b.Value.TotalSize);
            Compute.MatrixMultiply(a.Value.Data, tensor.Gradient.Data, gradB,
                a.Value.Shape[1], b.Value.Shape[1], a.Value.Shape[0], transposeA: true);
            Compute.Call(aidx, Compute.ElementwiseAddKernels, b.Gradient.Data, gradB, b.Gradient.Data);
        
            Compute.Return(gradA, gradB);
        }
    }

    public Value<float[]> ReadBias(BinaryReader reader) => new VectorValue(BinaryWriting.ReadVector(reader), AcceleratorIndex);
    public Value<float[,]> ReadWeights(BinaryReader reader) => new MatrixValue(BinaryWriting.ReadMatrix(reader), AcceleratorIndex);
    public void WriteBias(BinaryWriter writer, Value<float[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, Value<float[,]> weights) => BinaryWriting.WriteMatrix(writer, weights.ToHost());
}