using Jewels.Lazulite;
using Opal.NNs.Ff;
using Opal.NNs.Lstm;
using Opal.NNs.Recurrent;
using Opal.Utilities;

namespace Opal;

public class VectorCatalog : IFfCatalog<float[], float[], float[,], float[]>, IRecurrentCatalog<float[], float[], float[,]>, ILstmCatalog<float[], float[], float[,]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;

    public Tensor<float[]> Multiply(Tensor<float[,]> a, Tensor<float[]> b) => Operations.MatrixVectorMultiply(a, b);
    public Tensor<float[]> Add(Tensor<float[]> a, Tensor<float[]> b) => Operations.Add(a, b);


    public Tensor<float[]> ConcatHidden(Tensor<float[]> a, Tensor<float[]> b) => Operations.Concat(a, b);
    public Tensor<float[]> ConcatInputHidden(Tensor<float[]> a, Tensor<float[]> b) => Operations.Concat(a, b);

    public Value<float[]> ReadBias(BinaryReader reader) => new VectorValue(BinaryWriting.ReadVector(reader), AcceleratorIndex);
    public Value<float[,]> ReadWeights(BinaryReader reader) => new MatrixValue(BinaryWriting.ReadMatrix(reader), AcceleratorIndex);
    public void WriteBias(BinaryWriter writer, Value<float[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, Value<float[,]> weights) => BinaryWriting.WriteMatrix(writer, weights.ToHost());
    public Value<float[]> ReadState(BinaryReader reader) => new VectorValue(BinaryWriting.ReadVector(reader), AcceleratorIndex);
    public void WriteState(BinaryWriter writer, Value<float[]> state) => BinaryWriting.WriteVector(writer, state.ToHost());
}