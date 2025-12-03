using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static class LossFunctions
{
    private static Compute compute => Compute.Instance;
    
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] MeanSquaredErrorKernels { get; } = compute.Load((i, x, t, r) 
        => r[i] = (x[i] - t[i]) * (x[i] - t[i]));

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, float>[] MeanSquaredErrorBackwardKernels { get; } 
        = compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> r, float n) => 
            r[i] += grad[0] * 2 * (x[i] - t[i]) / n);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] CrossEntropyKernels { get; } 
        = compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> pred, 
            ArrayView1D<float, Stride1D.Dense> target, 
            ArrayView1D<float, Stride1D.Dense> r, 
            int size) =>
        {
            var p = XMath.Clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
            r[i] = -target[i] * XMath.Log(p) / size;
        });

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int>[] CrossEntropyBackwardKernels { get; } = compute.Load((
            Index1D i, 
            ArrayView1D<float, Stride1D.Dense> pred, 
            ArrayView1D<float, Stride1D.Dense> target, 
            ArrayView1D<float, Stride1D.Dense> grad, 
            ArrayView1D<float, Stride1D.Dense> r, int size) =>
        {
            var p = XMath.Clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
            r[i] += grad[0] * -target[i] / p / size;
        });
    
    public static Tensor<float> MeanSquaredError(ITensor predicted, IValue actual)
    {
        if (predicted.Value.TotalSize != actual.TotalSize)
            throw new ArgumentException("Values must be of the same length.");
        
        var aidx = predicted.Value.AcceleratorIndex;
        var result = compute.Get(aidx, 1);
        var (temp1, temp2) = (compute.Get(aidx, predicted.Value.TotalSize), compute.Get(aidx, 1));
        
        compute.Call(MeanSquaredErrorKernels, predicted.Value.Data, actual.Data, temp1);
        compute.Sum(temp1, temp2);
        compute.Call(compute.ElementwiseFloatMultiplyKernels, temp2, result, 1 / (float)actual.TotalSize);
        var val = new ScalarValue(result);
        compute.Return(temp1, temp2);
        
        return Operations.New(val, val.Zeros(), Backward, [predicted]);
        
        void Backward(ITensor t) =>
            compute.Call(
                MeanSquaredErrorBackwardKernels,
                predicted.Value.Data, actual.Data,
                t.Gradient.Data, predicted.Gradient.Data,
                actual.TotalSize);
    }
    
    private static Tensor<float> CrossEntropy(ITensor predicted, IValue actual, int size)
    {
        int aidx = predicted.Value.AcceleratorIndex;
        var result = compute.Get(aidx, 1);
        var (temp1, temp2) = (compute.Get(aidx, predicted.Value.TotalSize), compute.Get(aidx, 1));
    
        compute.Call(CrossEntropyKernels, predicted.Value.Data, actual.Data, temp1, size);
        compute.Sum(temp1, temp2);
        compute.Call(compute.ElementwiseFloatMultiplyKernels, temp2, result,  1f);
    
        compute.Return(temp1, temp2);
    
        return Operations.New(new ScalarValue(result), new ScalarValue(0f, aidx), Backward, [predicted]);
    
        void Backward(ITensor t) =>
            compute.Call(CrossEntropyBackwardKernels, predicted.Value.Data, actual.Data, t.Gradient.Data, predicted.Gradient.Data, size);
    }
    
    public static Func<ITensor, IValue, Tensor<float>> CreateCrossEntropy(int batchSize = 1) => (predicted, actual) => CrossEntropy(predicted, actual, batchSize);
}

