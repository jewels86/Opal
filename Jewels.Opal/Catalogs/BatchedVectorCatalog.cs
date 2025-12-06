using System.IO.Pipelines;
using Jewels.Lazulite;
using Jewels.Opal.NNs;
using Jewels.Opal.Utilities;

namespace Jewels.Opal;

public class BatchedVectorCatalog : IFfCatalog<float[,], float[,], float[,], float[]>, ILstmCatalog<float[,], float[,], float[,], float[]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;
    
    public Tensor<float[,]> Multiply(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.MatrixMultiply(b, a, transposeB: true);
    public Tensor<float[,]> Add(Tensor<float[]> a, Tensor<float[,]> b) => Operations.Add(b, a);
    public Tensor<float[,]> Add(Tensor<float[,]> a, Tensor<float[]> b) => Operations.Add(a, b);
    
    public Tensor<float[,]> ConcatHidden(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.Concat(a, b);
    public Tensor<float[,]> ConcatInputHidden(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.Concat(a, b);
    public Tensor<float[,]> LstmState(Tensor<float[,]> forgetGate, Tensor<float[,]> state, Tensor<float[,]> inputGate, Tensor<float[,]> cellGate) => 
        Operations.LstmState(forgetGate, state, inputGate, cellGate);
    public Tensor<float[,]> Sigmoid(Tensor<float[,]> x) => ActivationFunctions.Sigmoid(x);
    public Tensor<float[,]> Tanh(Tensor<float[,]> x) => ActivationFunctions.Tanh(x);

    public Value<float[]> ReadBias(BinaryReader reader) => Operations.New(BinaryWriting.ReadVector(reader), aidx: AcceleratorIndex);
    public Value<float[,]> ReadWeights(BinaryReader reader) => Operations.New(BinaryWriting.ReadMatrix(reader), aidx: AcceleratorIndex);
    public void WriteBias(BinaryWriter writer, Value<float[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, Value<float[,]> weight) => BinaryWriting.WriteMatrix(writer, weight.ToHost());
}