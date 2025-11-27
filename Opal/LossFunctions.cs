using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static class LossFunctions
{
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>> MeanSquaredErrorKernels { get; } = Compute.Load((i, x, t, r) 
        => r[i] = (x[i] - t[i]) * (x[i] - t[i]));

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, float>> MeanSquaredErrorBackwardKernels { get; } 
        = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> r, float n) => 
            r[i] += grad[i] * 2 * (x[i] - t[i]) / n);
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CrossEntropyKernels { get; } 
        = Compute.Load((i, pred, target, r) => r[i] = -target[i] * XMath.Log(pred[i]));

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, float>> CrossEntropyBackwardKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> pred, ArrayView1D<float, Stride1D.Dense> target, 
            ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> r, float n) => r[i] += grad[i] * -target[i] / pred[i] / n);
    
    public static Tensor<float> MeanSquaredError(ITensor predicted, IValue actual)
    {
        if (predicted.Value.TotalSize != actual.TotalSize)
            throw new ArgumentException("Vectors must be of the same length.");
        
        var aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, 1);
        var (temp1, temp2) = (Compute.GetTemp(aidx, predicted.Value.TotalSize), Compute.GetTemp(aidx, 1));
        Compute.Call(predicted.Value.AcceleratorIndex, MeanSquaredErrorKernels, predicted.Value.Data, actual.Data, temp1);
        Compute.Sum(temp1, temp2);
        Compute.Call(aidx, Compute.ElementwiseFloatMultiplyKernels, temp2, result, 1 / (float)actual.TotalSize);
        
        return Operations.New(new ScalarValue(result), new ScalarValue(0.0f, aidx), Backward, [predicted]);
        
        void Backward(ITensor t) =>
            Compute.Call(
                aidx, MeanSquaredErrorBackwardKernels,
                t.Gradient.Data.IntExtent,
                predicted.Value.Data, actual.Data,
                t.Gradient.Data, predicted.Gradient.Data,
                actual.TotalSize);
    }
    
    public static Tensor<float> CrossEntropy(ITensor predicted, IValue actual)
    {
        int aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, 1);
        var (temp1, temp2) = (Compute.Get(aidx, predicted.Value.TotalSize), Compute.Get(aidx, 1));
    
        Compute.Call(aidx, CrossEntropyKernels, predicted.Value.Data, actual.Data, temp1);
        Compute.Sum(temp1, temp2);
        Compute.Call(aidx, Compute.ElementwiseFloatMultiplyKernels, temp2, result,  1 / (float)actual.TotalSize);
    
        Compute.Return(temp1, temp2);
    
        return Operations.New(new ScalarValue(result), new ScalarValue(0.0f, aidx), Backward, [predicted]);
    
        void Backward(ITensor t) =>
            Compute.Call(aidx, CrossEntropyBackwardKernels, t.Gradient.Data.IntExtent,
                predicted.Value.Data, actual.Data, t.Gradient.Data, predicted.Gradient.Data, actual.TotalSize);
    }
}

