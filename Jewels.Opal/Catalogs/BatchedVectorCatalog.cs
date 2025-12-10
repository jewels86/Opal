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

    public (Tensor<float[,]>, Tensor<float[,]>) InLstmUpdate(Tensor<float[,]> input, Tensor<float[,]> hidden, Tensor<float[,]> state, LstmUpdateParameters<float[,], float[]> parameters)
    {
        var concat = ConcatInputHidden(input, hidden);
        var forgetWeighted = Multiply(parameters.ForgetWeights, concat);
        var inputWeighted = Multiply(parameters.InputWeights, concat);
        var cellWeighted = Multiply(parameters.CellWeights, concat);
        var outputWeighted = Multiply(parameters.OutputWeights, concat);

        KernelLstmUpdateParameters kernelParameters = new()
        {
            ForgetWeighted = forgetWeighted.Value,
            InputWeighted = inputWeighted.Value,
            CellWeighted = cellWeighted.Value,
            OutputWeighted = outputWeighted.Value,

            ForgetBias = parameters.ForgetBiases.Value,
            InputBias = parameters.InputBiases.Value,
            CellBias = parameters.CellBiases.Value,
            OutputBias = parameters.OutputBiases.Value,

            PrevState = state.Value,
        };
        
        var (newHidden, newState) = (Compute.GetLike(forgetWeighted.Value), Compute.GetLike(forgetWeighted.Value));
        Compute.Call(Operations.LstmUpdateKernels, newHidden, newState, kernelParameters);

        var newHiddenValue = hidden.Value.CreateAlike(newHidden);
        var newStateValue = state.Value.CreateAlike(newState);

        var newHiddenTensor = new Tensor<float[,]>(newHiddenValue, newHiddenValue.Zeros(), Backward, [
            input, hidden, state,
            parameters.ForgetWeights, parameters.InputWeights, parameters.CellWeights, parameters.OutputWeights,
            parameters.ForgetBiases, parameters.InputBiases, parameters.CellBiases, parameters.OutputBiases
        ]);
        var newStateTensor = new Tensor<float[,]>(newStateValue, newStateValue.Zeros());
        
        return (newHiddenTensor, newStateTensor);

        void Backward(ITensor tensor) // for state
        {
            KernelLstmUpdateBackwardParameters backwardParameters = new()
            {
                ForgetWeightGrad = parameters.ForgetWeights.Gradient,
                InputWeightGrad = parameters.InputWeights.Gradient,
                CellWeightGrad = parameters.CellWeights.Gradient,
                OutputWeightGrad = parameters.OutputWeights.Gradient,

                ForgetBiasGrad = parameters.ForgetBiases.Gradient,
                InputBiasGrad = parameters.InputBiases.Gradient,
                CellBiasGrad = parameters.CellBiases.Gradient,
                OutputBiasGrad = parameters.OutputBiases.Gradient,

                PrevStateGrad = state.Gradient,
            };
            Compute.Call(
                Operations.LstmUpdateBackwardKernels, 
                hidden.Gradient, state.Gradient, 
                tensor.Gradient.Data, kernelParameters, backwardParameters);
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
}

public static partial class Operations
{
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, KernelLstmUpdateParameters>[] LstmUpdateKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> hidden, 
            ArrayView1D<float, Stride1D.Dense> state, 
            KernelLstmUpdateParameters parameters) =>
        {
            var forgetGate = Sigmoid(parameters.ForgetWeighted[i] + parameters.ForgetBias[i]);
            var inputGate = Sigmoid(parameters.InputWeighted[i] + parameters.InputBias[i]);
            var cellGate = Tanh(parameters.CellWeighted[i] + parameters.CellBias[i]);
            var outputGate = Sigmoid(parameters.OutputWeighted[i] + parameters.OutputBias[i]);
            
            state[i] = (forgetGate * parameters.PrevState[i]) + (inputGate * cellGate);
            hidden[i] = outputGate * Tanh(state[i]);
        });
    
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, 
        KernelLstmUpdateParameters, 
        KernelLstmUpdateBackwardParameters>[] LstmUpdateBackwardKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> hiddenGrad, 
            ArrayView1D<float, Stride1D.Dense> stateGrad, 
            ArrayView1D<float, Stride1D.Dense> incomingGrad,
            KernelLstmUpdateParameters parameters, KernelLstmUpdateBackwardParameters backwardParameters) =>
        {
            var forgetGate = Sigmoid(parameters.ForgetWeighted[i] + parameters.ForgetBias[i]);
            var inputGate = Sigmoid(parameters.InputWeighted[i] + parameters.InputBias[i]);
            var cellGate = Tanh(parameters.CellWeighted[i] + parameters.CellBias[i]);
            var outputGate = Sigmoid(parameters.OutputWeighted[i] + parameters.OutputBias[i]);
            
            var state = (forgetGate * parameters.PrevState[i]) + (inputGate * cellGate);
            var tanhState = Tanh(state);
            
            stateGrad[i] += incomingGrad[i] * hiddenGrad[i] * outputGate * TanhBackward(outputGate);
            
            var forgetPart = stateGrad[i] * parameters.PrevState[i] * SigmoidBackward(forgetGate);
            var inputPart = stateGrad[i] * cellGate * SigmoidBackward(inputGate);
            var cellPart = stateGrad[i] * inputGate * TanhBackward(cellGate);
            var outputPart = hiddenGrad[i] * tanhState * SigmoidBackward(outputGate);
            var prevStatePart = stateGrad[i] * forgetGate;
            
            backwardParameters.ForgetWeightGrad[i] += forgetPart;
            backwardParameters.InputWeightGrad[i] += inputPart;
            backwardParameters.CellWeightGrad[i] += cellPart;
            backwardParameters.OutputWeightGrad[i] += outputPart;
            
            backwardParameters.ForgetBiasGrad[i] += forgetPart;
            backwardParameters.InputBiasGrad[i] += inputPart;
            backwardParameters.CellBiasGrad[i] += cellPart;
            backwardParameters.OutputBiasGrad[i] += outputPart;
            
            backwardParameters.PrevStateGrad[i] += prevStatePart;
        });

    
    public static float Sigmoid(float x) => 1f / (1f + XMath.Exp(-x));
    public static float Tanh(float x) => (float)Math.Tanh(x);
    
    public static float SigmoidBackward(float output) => output * (1f - output);
    public static float TanhBackward(float output) => 1f - (output * output);
}

public struct KernelLstmUpdateParameters
{
    public ArrayView1D<float, Stride1D.Dense> ForgetWeighted;
    public ArrayView1D<float, Stride1D.Dense> InputWeighted;
    public ArrayView1D<float, Stride1D.Dense> CellWeighted;
    public ArrayView1D<float, Stride1D.Dense> OutputWeighted;
    
    public ArrayView1D<float, Stride1D.Dense> ForgetBias;
    public ArrayView1D<float, Stride1D.Dense> InputBias;
    public ArrayView1D<float, Stride1D.Dense> CellBias;
    public ArrayView1D<float, Stride1D.Dense> OutputBias;
    
    public ArrayView1D<float, Stride1D.Dense> PrevState;
}

public struct KernelLstmUpdateBackwardParameters
{
    public ArrayView1D<float, Stride1D.Dense> ForgetWeightGrad;
    public ArrayView1D<float, Stride1D.Dense> InputWeightGrad;
    public ArrayView1D<float, Stride1D.Dense> CellWeightGrad;
    public ArrayView1D<float, Stride1D.Dense> OutputWeightGrad;
    
    public ArrayView1D<float, Stride1D.Dense> ForgetBiasGrad;
    public ArrayView1D<float, Stride1D.Dense> InputBiasGrad;
    public ArrayView1D<float, Stride1D.Dense> CellBiasGrad;
    public ArrayView1D<float, Stride1D.Dense> OutputBiasGrad;
    
    public ArrayView1D<float, Stride1D.Dense> PrevStateGrad;
}