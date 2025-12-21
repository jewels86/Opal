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

    public (Tensor<float[,]>, Tensor<float[,]>) InLstmUpdate(Tensor<float[,]> input, Tensor<float[,]> prevHidden, Tensor<float[,]> prevState, LstmUpdateParameters<float[,], float[]> parameters)
    {
        var concat = ConcatInputHidden(input, prevHidden);
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
        var newHiddenValue = prevHidden.Value.CreateAlike(newHidden);
        var newStateValue = prevState.Value.CreateAlike(newState);
        
        Console.WriteLine($"Lstm new hidden: {Operations.ToString(newHiddenValue.ToHost())}");

        List<ITensor> inputs =
        [
            prevState,
            forgetWeighted, inputWeighted, cellWeighted, outputWeighted,
            parameters.ForgetBiases, parameters.InputBiases, parameters.CellBiases, parameters.OutputBiases
        ];
        var newHiddenTensor = new Tensor<float[,]>(newHiddenValue, newHiddenValue.Zeros(), _ => {}, inputs);
        var newStateTensor = new Tensor<float[,]>(newStateValue, newStateValue.Zeros(), _ => {}, inputs);
        newHiddenTensor.BackwardAction = Backward;
        
        return (newHiddenTensor, newStateTensor);

        void Backward(ITensor tensor) // for hidden
        {
            var biasGrads = Compute.Get(forgetWeighted.AcceleratorIndex, forgetWeighted.TotalSize * 4);
            Compute.Call(LstmUpdateBackwardKernels, 
                newStateTensor.Gradient, prevState.Gradient, gatesOut, biasGrads,
                forgetWeighted.Gradient, inputWeighted.Gradient, cellWeighted.Gradient, outputWeighted.Gradient,
                newStateTensor.Value, prevState.Value, newHiddenTensor.Gradient);
            Compute.Call(LstmUpdateBackwardReductionKernels, 
                parameters.ForgetBiases.Gradient, parameters.InputBiases.Gradient, parameters.CellBiases.Gradient, parameters.OutputBiases.Gradient,
                biasGrads);
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
            hidden, state, gatesOut, prevState,
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
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateBackwardKernels { get; }
        = Compute.Load((
            i, stateGrad, prevStateGrad, gatesOut, biasGradsOut,
            forgetWeightedGrad, inputWeightedGrad, cellWeightedGrad, outputWeightedGrad,
            state, prevState, hiddenGrad) =>
        {
            var forgetGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4)];
            var inputGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 1)];
            var cellGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 2)];
            var outputGate = gatesOut[KernelProgramming.StridedIndexOf(i, 4, 3)];
            
            var tanhState = Operations.FloatTanh(state[i]);
            stateGrad[i] += hiddenGrad[i] * outputGate * Operations.FloatTanhBackward(tanhState);
            prevStateGrad[i] += stateGrad[i] * forgetGate;

            var forgetPart = stateGrad[i] * prevState[i] * Operations.FloatSigmoidBackward(forgetGate);
            var inputPart = stateGrad[i] * cellGate * Operations.FloatSigmoidBackward(inputGate);
            var cellPart = stateGrad[i] * inputGate * Operations.FloatTanhBackward(cellGate);
            var outputPart = hiddenGrad[i] * Operations.FloatTanh(state[i]) * Operations.FloatSigmoidBackward(outputGate);

            forgetWeightedGrad[i] += forgetPart;
            inputWeightedGrad[i] += inputPart;
            cellWeightedGrad[i] += cellPart;
            outputWeightedGrad[i] += outputPart;

            biasGradsOut[KernelProgramming.StridedIndexOf(i, 4)] += forgetPart;
            biasGradsOut[KernelProgramming.StridedIndexOf(i, 4, 1)] += inputPart;
            biasGradsOut[KernelProgramming.StridedIndexOf(i, 4, 2)] += cellPart;
            biasGradsOut[KernelProgramming.StridedIndexOf(i, 4, 3)] += outputPart;
        });

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] LstmUpdateBackwardReductionKernels { get; }
        = Compute.Load((
            i,
            forgetBiasGrad, inputBiasGrad, cellBiasGrad, outputBiasGrad,
            biasGradsIn) =>
        {
            var (forgetSum, inputSum, cellSum, outputSum) = (0f, 0f, 0f, 0f);
            var (size, num) = ((int)forgetBiasGrad.Length, (int)biasGradsIn.Length / (int)(4 * forgetBiasGrad.Length));

            for (int t = 0; t < num; t++)
            {
                int baseIndex = t * size + i;
                forgetSum += biasGradsIn[KernelProgramming.StridedIndexOf(baseIndex, 4)];
                inputSum += biasGradsIn[KernelProgramming.StridedIndexOf(baseIndex, 4, 1)];
                cellSum += biasGradsIn[KernelProgramming.StridedIndexOf(baseIndex, 4, 2)];
                outputSum += biasGradsIn[KernelProgramming.StridedIndexOf(baseIndex, 4, 3)];
            }

            forgetBiasGrad[i] += forgetSum;
            inputBiasGrad[i] += inputSum;
            cellBiasGrad[i] += cellSum;
            outputBiasGrad[i] += outputSum;
        });
}

public static partial class Operations
{
    public static float FloatSigmoid(float x) => 1f / (1f + XMath.Exp(-x));
    public static float FloatTanh(float x) => XMath.Tanh(x);
    
    public static float FloatSigmoidBackward(float output) => output * (1f - output);
    public static float FloatTanhBackward(float output) => 1f - (output * output);
}