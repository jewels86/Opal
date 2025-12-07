using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;
using Jewels.Opal.NNs;
using Jewels.Opal.Utilities;

namespace Jewels.Opal;

public class VectorCatalog : IFfCatalog<float[], float[], float[,], float[]>, IRecurrentCatalog<float[], float[], float[,]>, ILstmCatalog<float[], float[], float[,], float[]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;

    public Tensor<float[]> Multiply(Tensor<float[,]> a, Tensor<float[]> b) => Operations.MatrixVectorMultiply(a, b);
    public Tensor<float[]> Add(Tensor<float[]> a, Tensor<float[]> b) => Operations.Add(a, b);
    
    public Tensor<float[]> LstmState(Tensor<float[]> forgetGate, Tensor<float[]> state, Tensor<float[]> inputGate, Tensor<float[]> cellGate) => Operations.LstmState(forgetGate, state, inputGate, cellGate);
    public Tensor<float[]> ConcatHidden(Tensor<float[]> a, Tensor<float[]> b) => Operations.Concat(a, b);
    public Tensor<float[]> ConcatInputHidden(Tensor<float[]> a, Tensor<float[]> b) => Operations.Concat(a, b);
    public Tensor<float[]> LstmSigmoidGate(Tensor<float[]> weighted, Tensor<float[]> bias) => ActivationFunctions.Sigmoid(Operations.Add(weighted, bias));
    public Tensor<float[]> LstmTanhGate(Tensor<float[]> weighted, Tensor<float[]> bias) => ActivationFunctions.Tanh(Operations.Add(weighted, bias));
    public Tensor<float[]> LstmHidden(Tensor<float[]> outputGate, Tensor<float[]> newState) => Operations.Multiply(outputGate, ActivationFunctions.Tanh(newState));

    public Value<float[]> ReadBias(BinaryReader reader) => new VectorValue(BinaryWriting.ReadVector(reader), AcceleratorIndex);
    public Value<float[,]> ReadWeights(BinaryReader reader) => new MatrixValue(BinaryWriting.ReadMatrix(reader), AcceleratorIndex);
    public void WriteBias(BinaryWriter writer, Value<float[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, Value<float[,]> weights) => BinaryWriting.WriteMatrix(writer, weights.ToHost());
    public Value<float[]> ReadState(BinaryReader reader) => new VectorValue(BinaryWriting.ReadVector(reader), AcceleratorIndex);
    public void WriteState(BinaryWriter writer, Value<float[]> state) => BinaryWriting.WriteVector(writer, state.ToHost());
}

public static partial class Operations
{
    #region Kernels
    //public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> LstmSigmoidGateKernels { get; }
   //     = Compute.Load((i, weighted, bias, result) => )
   // ill do this later
    #endregion
}