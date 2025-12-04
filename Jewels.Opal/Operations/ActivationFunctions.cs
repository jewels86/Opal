using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

public static class ActivationFunctions
{
    private static Compute compute => Compute.Instance;
    
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ReLuBackwardKernels { get; }
        = compute.Load((i, x, grad, r) => r[i] += x[i] > 0 ? grad[i] : 0.0f);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SigmoidKernels { get; } 
        = compute.Load((i, x, r) => r[i] = 1 / (1 + XMath.Exp(-x[i])));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SigmoidBackwardKernels { get; } 
        = compute.Load((i, x, grad, r) => r[i] += grad[i] * x[i] * (1 - x[i]));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] TanhBackwardKernels { get; } 
        = compute.Load((i, x, grad, r) => r[i] += grad[i] * (1 - x[i] * x[i]));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] AccumulateGradientKernels { get; } 
        = compute.Load((i, grad, r) => r[i] += grad[i]);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] SoftmaxBackwardKernels { get; }
        = compute.Load((i, grad, softmax, dot, r) => r[i] += softmax[i] * (grad[i] - dot[0]));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedSumRowKernels { get; } = compute.Load((Index1D batchIdx, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> sums, int numClasses) =>
    {
        var (sum, offset) = (0f, batchIdx * numClasses);
        for (int i = 0; i < numClasses; i++)
            sum += data[offset + i];
        sums[batchIdx] = sum;
    });
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedDivideByRowSumKernels { get; } = compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> sums, ArrayView1D<float, Stride1D.Dense> result, int numClasses) =>
    {
        int batchIdx = i / numClasses;
        result[i] = data[i] / sums[batchIdx];
    });
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedSoftmaxBackwardKernels { get; } = compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> softmax, ArrayView1D<float, Stride1D.Dense> dot, ArrayView1D<float, Stride1D.Dense> result, int numClasses) =>
    {
        int batchIdx = i / numClasses;
        result[i] += softmax[i] * (grad[i] - dot[batchIdx]);
    });
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedMaxRowKernels { get; } = compute.Load((Index1D batchIdx, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> maxs, int numClasses) =>
    {
        float max = float.MinValue;
        int offset = batchIdx * numClasses;
        for (int i = 0; i < numClasses; i++)
            max = XMath.Max(max, data[offset + i]);
        maxs[batchIdx] = max;
    });

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedExpWithMaxKernels { get; } = compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> maxs, ArrayView1D<float, Stride1D.Dense> result, int numClasses) =>
    {
        int batchIdx = i / numClasses;
        result[i] = XMath.Exp(x[i] - maxs[batchIdx]);
    });

    
    public static ITensor ActivationFunction(
        ITensor x, 
        Action<int, MemoryBuffer1D<float, Stride1D.Dense>> forward, 
        Func<int, MemoryBuffer1D<float, Stride1D.Dense>, Action<ITensor>> backward)
    {
        int aidx = x.Value.AcceleratorIndex;
        var result = compute.Get(aidx, x.Value.TotalSize);
        forward(aidx, result);
        return x.Create(x.Value.CreateAlike(result), x.Gradient.Zeros(), backward(aidx, result), [x]);
    }
    
    public static ITensor ReLu(ITensor x) => ActivationFunction(x, 
            (_, r) => compute.Call(compute.ElementwiseFloatMaxKernels, x.Value.Data, r, 0.0f), 
            (_, _) => t => compute.Call(ReLuBackwardKernels, x.Value.Data, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Sigmoid(ITensor x) => ActivationFunction(x, 
        (_, r) => compute.Call(SigmoidKernels, x.Value.Data, r),
            (_, r) => t => compute.Call(SigmoidBackwardKernels, r, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Tanh(ITensor x) =>
        ActivationFunction(x, (_, r) => compute.Call(compute.ElementwiseTanhKernels, x.Value.Data, r),
            (_, r) => t => compute.Call(TanhBackwardKernels, r, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Identity(ITensor x) =>
        ActivationFunction(x, (_, r) => compute.Call(compute.CopyKernels, x.Value.Data, r), 
            (_, _) => t => compute.Call(AccumulateGradientKernels, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Softmax(ITensor x) =>
        ActivationFunction(x, (a, r) =>
        {
            var temps = compute.Get(a, 2, x.Value.TotalSize);
            compute.Call(compute.ElementwiseExpKernels, x.Value.Data, temps[0]);
            compute.Sum(temps[0], temps[1]);
            compute.Call(compute.ElementwiseScalarDivideKernels, temps[0], temps[1], r);
            compute.Return(temps);
        }, (a, r) => t =>
        {
            var temp = compute.Get(a, x.Value.TotalSize);
            compute.Dot(t.Gradient.Data, r, temp);
            compute.Call(SoftmaxBackwardKernels, t.Gradient.Data, r, temp, x.Gradient.Data);
            compute.Return(temp);
        });

    public static Tensor<float[,]> BatchedSoftmax(Tensor<float[,]> x)
    {
        var (aidx, batchSize, numClasses, totalSize) = (x.Value.AcceleratorIndex, x.Value.Shape[0], x.Value.Shape[1], x.Value.TotalSize);
        var (maxs, exp, sums, result) = 
            (compute.Get(aidx, batchSize), compute.Get(aidx, totalSize), compute.Get(aidx, batchSize), compute.Get(aidx, totalSize));
    
        compute.Call(BatchedMaxRowKernels, x.Value, maxs, numClasses);
        compute.Call(BatchedExpWithMaxKernels, x.Value, maxs, exp, numClasses);
        compute.Call(BatchedSumRowKernels, exp, sums, numClasses);
        compute.Call(BatchedDivideByRowSumKernels, exp, sums, result, numClasses);
    
        compute.Return(exp, sums, maxs);
        
        compute.Synchronize(aidx);
        if (exp.GetAsArray1D().Any(float.IsNaN)) throw new InvalidOperationException("Softmax contains NaN values.");
    
        return x.Create(x.Value.CreateAlike(result), x.Gradient.Zeros(), Backward, [x]);
    
        void Backward(ITensor t)
        {
            var temp = compute.Get(aidx, totalSize);
            var dot = compute.Get(aidx, batchSize);
        
            compute.Call(compute.ElementwiseMultiplyKernels, t.Gradient.Data, result, temp);
            compute.Call(BatchedSumRowKernels, temp, dot, numClasses);
            compute.Call(BatchedSoftmaxBackwardKernels, t.Gradient.Data, result, dot, x.Gradient.Data, numClasses);
        
            compute.Return(temp, dot);
        }
    }
    
    #region Overloads
    public static Tensor<T> ReLu<T>(Tensor<T> x) where T : notnull => (Tensor<T>)ReLu(x as ITensor);
    public static Tensor<T> Sigmoid<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Sigmoid(x as ITensor);
    public static Tensor<T> Tanh<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Tanh(x as ITensor);
    public static Tensor<T> Identity<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Identity(x as ITensor);
    public static Tensor<T> Softmax<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Softmax(x as ITensor);
    #endregion
}
