using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

public static class LossFunctions
{
    /// <summary>
    /// (r, x, t) => r[i] = (x[i] - t[i]) * (x[i] - t[i])
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] MeanSquaredErrorKernels { get; } = Compute.Load((i, r, x, t) 
        => r[i] = (x[i] - t[i]) * (x[i] - t[i]));

    /// <summary>
    /// (r, x, t, grad, n) => r[i] += grad[0] * 2 * (x[i] - t[i[) / n
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, float>[] MeanSquaredErrorBackwardKernels { get; } 
        = Compute.Load((Index1D i, 
                ArrayView1D<float, Stride1D.Dense> r,
                ArrayView1D<float, Stride1D.Dense> x, 
                ArrayView1D<float, Stride1D.Dense> t, 
                ArrayView1D<float, Stride1D.Dense> grad, float n) => 
            r[i] += grad[0] * 2 * (x[i] - t[i]) / n);
    
    /// <summary>
    /// (r, pred, target, size) => r = -target * log(pred) / size
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] CrossEntropyKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> r, 
            ArrayView1D<float, Stride1D.Dense> pred, 
            ArrayView1D<float, Stride1D.Dense> target, 
            int size) =>
        {
            var p = XMath.Clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
            r[i] = -target[i] * XMath.Log(p) / size;
        });

    /// <summary>
    /// (r, pred, target, grad, size) => r[i] += grad[0] * -target[i] / pred[i] / size
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, int>[] CrossEntropyBackwardKernels { get; } = Compute.Load((
            Index1D i, 
            ArrayView1D<float, Stride1D.Dense> r, 
            ArrayView1D<float, Stride1D.Dense> pred, 
            ArrayView1D<float, Stride1D.Dense> target, 
            ArrayView1D<float, Stride1D.Dense> grad, 
            int size) =>
        {
            var p = XMath.Clamp(pred[i], 1e-7f, 1.0f - 1e-7f);
            r[i] += grad[0] * -target[i] / p / size;
        });
    
    public static Tensor<float> MeanSquaredError(ITensor predicted, IValue actual)
    {
        if (predicted.Value.TotalSize != actual.TotalSize)
            throw new ArgumentException($"Values must be of the same length- shapes {Operations.ToString(predicted.Value.Shape)} vs {Operations.ToString(actual.Shape)} (sizes {predicted.Value.TotalSize} vs {actual.TotalSize})");
        
        var aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, 1);
        var (temp1, temp2) = (Compute.Get(aidx, predicted.Value.TotalSize), Compute.Get(aidx, 1));
        
        Compute.Call(MeanSquaredErrorKernels, temp1, predicted.Value.Data, actual.Data);
        Compute.Sum(temp2, temp1);
        Compute.Call(Compute.FloatMultiplyKernels, result, temp2, 1 / (float)actual.TotalSize);
        var val = new ScalarValue(result);
        Compute.Return(temp1, temp2);
        
        return Operations.New(val, val.Zeros(), Backward, [predicted]);
        
        void Backward(ITensor t) =>
            Compute.Call(
                MeanSquaredErrorBackwardKernels,
                predicted.Gradient.Data, predicted.Value.Data, actual.Data,
                t.Gradient.Data,
                actual.TotalSize);
    }
    
    private static Tensor<float> CrossEntropy(ITensor predicted, IValue actual, int size)
    {
        int aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, 1);
        var (temp1, temp2) = (Compute.Get(aidx, predicted.Value.TotalSize), Compute.Get(aidx, 1));
    
        Compute.Call(CrossEntropyKernels, temp1, predicted.Value.Data, actual.Data, size);
        Compute.Sum(temp2, temp1);
        Compute.Call(Compute.FloatMultiplyKernels, result, temp2, 1f);
    
        Compute.Return(temp1, temp2);
    
        return Operations.New(new ScalarValue(result), new ScalarValue(0f, aidx), Backward, [predicted]);
    
        void Backward(ITensor t) =>
            Compute.Call(CrossEntropyBackwardKernels, predicted.Gradient.Data, predicted.Value.Data, actual.Data, t.Gradient.Data, size);
    }
    
    public static Func<ITensor, IValue, Tensor<float>> CreateCrossEntropy(int batchSize = 1) => (predicted, actual) => CrossEntropy(predicted, actual, batchSize);
}

