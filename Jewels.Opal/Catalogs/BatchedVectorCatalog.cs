using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;
using Jewels.Opal.NNs;
using Jewels.Opal.Utilities;

namespace Jewels.Opal;

public class BatchedVectorCatalog : 
    IFfCatalog<float[,], float[,], float[,], float[]>, 
    ILstmCatalog<float[,], float[,], float[,], float[]>, 
    IOptimizedLstmCatalog<float[,], float[,], float[,], float[]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;
    public bool Production { get; set; } = false;
    
    public Tensor<float[,]> Multiply(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.MatrixMultiply(b, a, transposeB: true);
    public Tensor<float[,]> Add(Tensor<float[]> a, Tensor<float[,]> b) => Operations.Add(b, a);
    
    public Tensor<float[,]> ConcatHidden(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.Concat(a, b);
    public Tensor<float[,]> ConcatInputHidden(Tensor<float[,]> a, Tensor<float[,]> b) => Operations.Concat(a, b);
    public Tensor<float[,]> LstmState(Tensor<float[,]> forgetGate, Tensor<float[,]> state, Tensor<float[,]> inputGate, Tensor<float[,]> cellGate) => 
        Operations.LstmState(forgetGate, state, inputGate, cellGate);

    public Tensor<float[,]> LstmSigmoidGate(Tensor<float[,]> weighted, Tensor<float[]> bias) => ActivationFunctions.Sigmoid(Operations.Add(weighted, bias));
    public Tensor<float[,]> LstmTanhGate(Tensor<float[,]> weighted, Tensor<float[]> bias) => ActivationFunctions.Tanh(Operations.Add(weighted, bias));
    public Tensor<float[,]> LstmHidden(Tensor<float[,]> outputGate, Tensor<float[,]> newState) => Operations.Multiply(outputGate, ActivationFunctions.Tanh(newState));

    public (Tensor<float[,]>, Tensor<float[,]>) InLstmUpdate(Tensor<float[,]> input, Tensor<float[,]> hidden, Tensor<float[,]> prevState, LstmUpdateParameters<float[,], float[]> parameters)
    {
        var concat = ConcatInputHidden(input, hidden);
        var forgetWeighted = Multiply(parameters.ForgetWeights, concat);
        var inputWeighted = Multiply(parameters.InputWeights, concat);
        var cellWeighted = Multiply(parameters.CellWeights, concat);
        var outputWeighted = Multiply(parameters.OutputWeights, concat);
        
        var (newHidden, newState) = (Compute.GetLike(forgetWeighted.Value), Compute.GetLike(forgetWeighted.Value));
        var gatesOut = Production ? Compute.Get(forgetWeighted.AcceleratorIndex, 0) : Compute.Get(forgetWeighted.AcceleratorIndex, forgetWeighted.TotalSize * 4);
        Compute.Call(LstmUpdateKernels,
            newHidden, newState, gatesOut, prevState.Value,
            forgetWeighted.Value, inputWeighted.Value, cellWeighted.Value, outputWeighted.Value,
            parameters.ForgetBiases.Value, parameters.InputBiases.Value, parameters.CellBiases.Value, parameters.OutputBiases.Value);
        
        if (Production) gatesOut.Dispose();
        var newHiddenValue = hidden.Value.CreateAlike(newHidden);
        var newStateValue = prevState.Value.CreateAlike(newState);

        List<ITensor> inputs =
        [
            input, hidden, prevState,
            forgetWeighted, inputWeighted, cellWeighted, outputWeighted,
            parameters.ForgetWeights, parameters.InputWeights, parameters.CellWeights, parameters.OutputWeights,
            parameters.ForgetBiases, parameters.InputBiases, parameters.CellBiases, parameters.OutputBiases
        ];
        var newHiddenTensor = new Tensor<float[,]>(newHiddenValue, newHiddenValue.Zeros(), _ => {}, inputs);
        var newStateTensor = new Tensor<float[,]>(newStateValue, newStateValue.Zeros(), _ => {}, inputs);
        newHiddenTensor.BackwardAction = Backward;
        
        return (newHiddenTensor, newStateTensor);

        void Backward(ITensor tensor) // for hidden
        {
            Compute.Call(LstmUpdateBackwardKernels, 
                newStateTensor.Gradient, prevState.Gradient, gatesOut,
                forgetWeighted.Gradient, inputWeighted.Gradient, cellWeighted.Gradient, outputWeighted.Gradient,
                parameters.ForgetBiases.Gradient, parameters.InputBiases.Gradient, parameters.CellBiases.Gradient, parameters.OutputBiases.Gradient,
                newStateTensor.Value, prevState.Value, hidden.Gradient);
            gatesOut.Return();
        }
    }

    public (Tensor<float[,]>, Tensor<float[,]>) OutLstmUpdate(Tensor<float[,]> input, Tensor<float[,]> hidden, Tensor<float[,]> state, LstmUpdateParameters<float[,], float[]> parameters)
    {
        return InLstmUpdate(input, hidden, state, parameters);
    }

    public Value<float[]> ReadBias(BinaryReader reader) => Operations.New(BinaryWriting.ReadVector(reader), aidx: AcceleratorIndex);
    public Value<float[,]> ReadWeights(BinaryReader reader) => Operations.New(BinaryWriting.ReadMatrix(reader), aidx: AcceleratorIndex);
    public void WriteBias(BinaryWriter writer, Value<float[]> bias) => BinaryWriting.WriteVector(writer, bias.ToHost());
    public void WriteWeights(BinaryWriter writer, Value<float[,]> weight) => BinaryWriting.WriteMatrix(writer, weight.ToHost());
    
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateKernels { get; }
    = Compute.Load((i,
        hidden, state, gatesOut,  prevState, 
        forgetWeighted, inputWeighted, cellWeighted, outputWeighted,
        forgetBias, inputBias, cellBias, outputBias) =>
    {
        var featureIndex = i % forgetBias.Length;
        var forgetGate = Operations.FloatSigmoid(forgetWeighted[i] + forgetBias[featureIndex]);
        var inputGate = Operations.FloatSigmoid(inputWeighted[i] + inputBias[featureIndex]);
        var cellGate = Operations.FloatTanh(cellWeighted[i] + cellBias[featureIndex]);
        var outputGate = Operations.FloatSigmoid(outputWeighted[i] + outputBias[featureIndex]);
        state[i] = forgetGate * prevState[i] + inputGate * cellGate;
        hidden[i] = outputGate * Operations.FloatTanh(state[i]);

        if (gatesOut.IntLength == 0) return;
        
        gatesOut[KernelProgramming.StridedIndexOf(i, 4)] = forgetGate;
        gatesOut[KernelProgramming.StridedIndexOf(i, 4, 1)] = inputGate;
        gatesOut[KernelProgramming.StridedIndexOf(i, 4, 2)] = cellGate;
        gatesOut[KernelProgramming.StridedIndexOf(i, 4, 3)] = outputGate;
    });

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateBackwardKernels { get; }
    = Compute.Load((
        i, stateGrad, prevStateGrad, gatesOut,
        forgetWeightedGrad, inputWeightedGrad, cellWeightedGrad, outputWeightedGrad,
        forgetBiasGrad, inputBiasGrad, cellBiasGrad, outputBiasGrad,
        state, prevState, hiddenGrad) =>
    {
        var forgetGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4)];
        var inputGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 1)];
        var cellGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 2)];
        var outputGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 3)];
        
        stateGrad[i] += hiddenGrad[i] * outputGate * Operations.FloatTanhBackward(state[i]);
        prevStateGrad[i] += stateGrad[i] * forgetGate;
        
        var forgetPart = stateGrad[i] * prevState[i] * Operations.FloatSigmoidBackward(forgetGate);
        var inputPart = stateGrad[i] * cellGate * Operations.FloatSigmoidBackward(inputGate);
        var cellPart = stateGrad[i] * inputGate * Operations.FloatTanhBackward(cellGate);
        var outputPart = hiddenGrad[i] * Operations.FloatTanh(state[i]) * Operations.FloatSigmoidBackward(outputGate);
        
        forgetWeightedGrad[i] += forgetPart;
        inputWeightedGrad[i] += inputPart;
        cellWeightedGrad[i] += cellPart;
        outputWeightedGrad[i] += outputPart;

        var featureIndex = i % inputBiasGrad.Length;
        Atomic.Add(ref forgetBiasGrad[featureIndex], forgetPart);
        Atomic.Add(ref inputBiasGrad[featureIndex], inputPart);
        Atomic.Add(ref cellBiasGrad[featureIndex], cellPart);
        Atomic.Add(ref outputBiasGrad[featureIndex], outputPart); 
        // later, this can be replaced with a strided array and a reduction pass
    });
}

public static partial class Operations
{
    public static float FloatSigmoid(float x) => 1f / (1f + XMath.Exp(-x));
    public static float FloatTanh(float x) => (float)Math.Tanh(x);
    
    public static float FloatSigmoidBackward(float output) => output * (1f - output);
    public static float FloatTanhBackward(float output) => 1f - (output * output);
}