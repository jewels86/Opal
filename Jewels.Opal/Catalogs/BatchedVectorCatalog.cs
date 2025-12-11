using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;
using Jewels.Opal.NNs;
using Jewels.Opal.Utilities;

namespace Jewels.Opal;

public class BatchedVectorCatalog : IFfCatalog<float[,], float[,], float[,], float[]>, ILstmCatalog<float[,], float[,], float[,], float[]>, IOptimizedLstmCatalog<float[,], float[,], float[,], float[]>
{
    public int AcceleratorIndex { get; set; } = Operations.DefaultAcceleratorIndex;
    
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
        Compute.Call(LstmUpdateKernels,
            newHidden, newState, prevState.Value,
            forgetWeighted.Value, inputWeighted.Value, cellWeighted.Value, outputWeighted.Value,
            parameters.ForgetBiases.Value, parameters.InputBiases.Value, parameters.CellBiases.Value, parameters.OutputBiases.Value);

        var newHiddenValue = hidden.Value.CreateAlike(newHidden);
        var newStateValue = prevState.Value.CreateAlike(newState);

        var newHiddenTensor = new Tensor<float[,]>(newHiddenValue, newHiddenValue.Zeros(), _ => {}, [
            input, hidden, prevState,
            parameters.ForgetWeights, parameters.InputWeights, parameters.CellWeights, parameters.OutputWeights,
            parameters.ForgetBiases, parameters.InputBiases, parameters.CellBiases, parameters.OutputBiases
        ]);
        var newStateTensor = new Tensor<float[,]>(newStateValue, newStateValue.Zeros());
        newHiddenTensor.BackwardAction = Backward;
        
        return (newHiddenTensor, newStateTensor);

        void Backward(ITensor tensor) // for hidden
        {
            Compute.Call(LstmUpdateBackward1Kernels,
                newStateTensor.Gradient, forgetWeighted.Value, inputWeighted.Value, cellWeighted.Value,
                parameters.ForgetBiases.Value, parameters.InputBiases.Value, parameters.CellBiases.Value,
                inputWeighted.Gradient, cellWeighted.Gradient,
                parameters.InputBiases.Gradient, parameters.CellBiases.Gradient,
                prevState.Value, newHiddenTensor.Gradient);
            Compute.Call(LstmUpdateBackward2Kernels,
                newStateTensor.Gradient, prevState.Gradient,
                forgetWeighted.Value, outputWeighted.Value,
                parameters.ForgetBiases.Value, parameters.OutputBiases.Value,
                forgetWeighted.Gradient, outputWeighted.Gradient,
                parameters.ForgetBiases.Gradient, parameters.OutputBiases.Gradient,
                prevState.Value, hidden.Value, newHiddenTensor.Gradient);
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
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateKernels { get; }
    = Compute.Load((i,
        hidden, state, prevState, 
        forgetWeighted, inputWeighted, cellWeighted, outputWeighted,
        forgetBias, inputBias, cellBias, outputBias) =>
    {
        var featureIndex = i % forgetBias.Length;
        var forgetGate = Operations.Sigmoid(forgetWeighted[i] + forgetBias[featureIndex]);
        var inputGate = Operations.Sigmoid(inputWeighted[i] + inputBias[featureIndex]);
        var cellGate = Operations.Tanh(cellWeighted[i] + cellBias[featureIndex]);
        var outputGate = Operations.Sigmoid(outputWeighted[i] + outputBias[featureIndex]);
        state[i] = forgetGate * prevState[i] + inputGate * cellGate;
        hidden[i] = outputGate * Operations.Tanh(state[i]);
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
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateBackward1Kernels { get; }
    = Compute.Load((
        i, stateGrad, 
        forgetWeighted, inputWeighted, cellWeighted,
        forgetBias, inputBias, cellBias,
        inputWeightedGrad, cellWeightedGrad, 
        inputBiasGrad, cellBiasGrad,
        prevState, hiddenGrad) =>
    {
        var featureIndex = i % forgetBias.Length;
        var forgetGate = Operations.Sigmoid(forgetWeighted[i] + forgetBias[featureIndex]);
        var inputGate = Operations.Sigmoid(inputWeighted[i] + inputBias[featureIndex]);
        var cellGate = Operations.Tanh(cellWeighted[i] + cellBias[featureIndex]);
        var state = forgetGate * prevState[i] + inputGate * cellGate;

        stateGrad[i] += hiddenGrad[i] * Operations.TanhBackward(state);
        
        var inputPart = stateGrad[i] * cellGate * Operations.SigmoidBackward(inputGate);
        var cellPart = stateGrad[i] * inputGate * Operations.TanhBackward(cellGate);
        
        inputWeightedGrad[i] += inputPart;
        cellWeightedGrad[i] += cellPart;

        Atomic.Add(ref inputBiasGrad[featureIndex], inputPart);
        Atomic.Add(ref cellBiasGrad[featureIndex], cellPart);
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
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateBackward2Kernels { get; }
    = Compute.Load((
        i, stateGrad, prevStateGrad,
        forgetWeighted, outputWeighted,
        forgetBias, outputBias,
        forgetWeightedGrad, outputWeightedGrad, 
        forgetBiasGrad, outputBiasGrad,
        prevState, state, hiddenGrad) =>
    {
        var featureIndex = i % forgetBias.Length;
        var forgetGate = Operations.Sigmoid(forgetWeighted[i] + forgetBias[featureIndex]);
        var outputGate = Operations.Sigmoid(outputWeighted[i] + outputBias[featureIndex]);
        
        var forgetPart = stateGrad[i] * prevState[i] * Operations.SigmoidBackward(forgetGate);
        var outputPart = hiddenGrad[i] * Operations.Tanh(state[i]) * Operations.SigmoidBackward(outputGate);
        
        forgetWeightedGrad[i] += forgetPart;
        outputWeightedGrad[i] += outputPart;
        
        Atomic.Add(ref forgetBiasGrad[featureIndex], forgetPart);
        Atomic.Add(ref outputBiasGrad[featureIndex], outputPart);
        
        prevStateGrad[i] += stateGrad[i] * forgetGate;
    });
}

public static partial class Operations
{
    public static float Sigmoid(float x) => 1f / (1f + XMath.Exp(-x));
    public static float Tanh(float x) => (float)Math.Tanh(x);
    
    public static float SigmoidBackward(float output) => output * (1f - output);
    public static float TanhBackward(float output) => 1f - (output * output);
}