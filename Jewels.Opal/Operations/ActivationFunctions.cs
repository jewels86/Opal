using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

public static class ActivationFunctions
{
    /// <summary>
    /// (r, x, grad) => r += x > 0 ? grad : 0
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ReLuBackwardKernels { get; }
        = Compute.Load((i, r, x, grad) => r[i] += x[i] > 0 ? grad[i] : 0.0f);
    
    /// <summary>
    /// (r, x) => r = 1 / (1 + e^(-x))
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SigmoidKernels { get; } 
        = Compute.Load((i, r, x) => r[i] = 1 / (1 + XMath.Exp(-x[i])));
    
    /// <summary>
    /// (r, x, grad) => r += grad * x * (1 - x)
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SigmoidBackwardKernels { get; } 
        = Compute.Load((i, r, x, grad) => r[i] += grad[i] * x[i] * (1 - x[i]));
    
    /// <summary>
    /// (r, grad) => r += grad
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] AccumulateGradientKernels { get; } 
        = Compute.Load((i, r, grad) => r[i] += grad[i]);

    /// <summary>
    /// (r, grad, softmax, dot) => r += softmax * (grad - dot[0])
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] SoftmaxBackwardKernels { get; }
        = Compute.Load((i, r, grad, softmax, dot) => r[i] += softmax[i] * (grad[i] - dot[0]));
    
    /// <summary>
    /// (sums, data, numClasses) => sums[batchIdx] = sum of row
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedSumRowKernels { get; } = Compute.Load((Index1D batchIdx, ArrayView1D<float, Stride1D.Dense> sums, ArrayView1D<float, Stride1D.Dense> data, int numClasses) =>
    {
        var (sum, offset) = (0f, batchIdx * numClasses);
        for (int i = 0; i < numClasses; i++)
            sum += data[offset + i];
        sums[batchIdx] = sum;
    });
    
    /// <summary>
    /// (result, data, sums, numClasses) => result[i] = data[i] / sums[batchIdx]
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedDivideByRowSumKernels { get; } = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> data, ArrayView1D<float, Stride1D.Dense> sums, int numClasses) =>
    {
        int batchIdx = i / numClasses;
        result[i] = data[i] / sums[batchIdx];
    });
    
    /// <summary>
    /// (result, grad, softmax, dot, numClasses) => result[i] += softmax[i] * (grad[i] - dot[batchIdx])
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedSoftmaxBackwardKernels { get; } = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> grad, ArrayView1D<float, Stride1D.Dense> softmax, ArrayView1D<float, Stride1D.Dense> dot, int numClasses) =>
    {
        int batchIdx = i / numClasses;
        result[i] += softmax[i] * (grad[i] - dot[batchIdx]);
    });
    
    /// <summary>
    /// (maxs, data, numClasses) => maxs[batchIdx] = max of row
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedMaxRowKernels { get; } = Compute.Load((Index1D batchIdx, ArrayView1D<float, Stride1D.Dense> maxs, ArrayView1D<float, Stride1D.Dense> data, int numClasses) =>
    {
        float max = float.MinValue;
        int offset = batchIdx * numClasses;
        for (int i = 0; i < numClasses; i++)
            max = XMath.Max(max, data[offset + i]);
        maxs[batchIdx] = max;
    });

    /// <summary>
    /// (result, x, maxs, numClasses) => result[i] = e^(x[i] - maxs[batchIdx])
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[] 
        BatchedExpWithMaxKernels { get; } = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> maxs, int numClasses) =>
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
        var result = Compute.Get(aidx, x.Value.TotalSize);
        forward(aidx, result);
        return x.Create(x.Value.CreateAlike(result), x.Gradient.Zeros(), backward(aidx, result), [x]);
    }
    
    public static ITensor ReLu(ITensor x) => ActivationFunction(x, 
            (_, r) => Compute.FloatMax(r, x.Value.Data, 0.0f), 
            (_, _) => t => Compute.Call(ReLuBackwardKernels, x.Gradient.Data, x.Value.Data, t.Gradient.Data));

    public static ITensor Sigmoid(ITensor x) => ActivationFunction(x, 
        (_, r) => Compute.Call(SigmoidKernels, r, x.Value.Data),
            (_, r) => t => Compute.Call(SigmoidBackwardKernels, x.Gradient.Data, r, t.Gradient.Data));

    public static ITensor Tanh(ITensor x) =>
        ActivationFunction(x, (_, r) => Compute.Call(Operations.TanhKernels, r, x.Value.Data),
            (_, r) => t => Compute.Call(Operations.TanhBackwardKernels, x.Gradient.Data, r, t.Gradient.Data));

    public static ITensor Identity(ITensor x) =>
        ActivationFunction(x, (_, r) => Compute.Call(Compute.CopyKernels, r, x.Value.Data), 
            (_, _) => t => Compute.Call(AccumulateGradientKernels, x.Gradient.Data, t.Gradient.Data));

    public static ITensor Softmax(ITensor x) =>
        ActivationFunction(x, (a, r) =>
        {
            var temps = Compute.Get(a, 2, x.Value.TotalSize);
            Compute.Call(Compute.ExpKernels, temps[0], x.Value.Data);
            Compute.Sum(temps[0], temps[1]);
            Compute.Call(Compute.ScalarDivideKernels, r, temps[0], temps[1]);
            Compute.Return(temps);
        }, (a, r) => t =>
        {
            var temp = Compute.Get(a, x.Value.TotalSize);
            Compute.Dot(temp, t.Gradient.Data, r);
            Compute.Call(SoftmaxBackwardKernels, x.Gradient.Data, t.Gradient.Data, r, temp);
            Compute.Return(temp);
        });

    public static Tensor<float[,]> BatchedSoftmax(Tensor<float[,]> x)
    {
        var (aidx, batchSize, numClasses, totalSize) = (x.Value.AcceleratorIndex, x.Value.Shape[0], x.Value.Shape[1], x.Value.TotalSize);
        var (maxs, exp, sums, result) = 
            (Compute.Get(aidx, batchSize), Compute.Get(aidx, totalSize), Compute.Get(aidx, batchSize), Compute.Get(aidx, totalSize));
    
        Compute.Call(BatchedMaxRowKernels, maxs, x.Value, numClasses);
        Compute.Call(BatchedExpWithMaxKernels, exp, x.Value, maxs, numClasses);
        Compute.Call(BatchedSumRowKernels, sums, exp, numClasses);
        Compute.Call(BatchedDivideByRowSumKernels, result, exp, sums, numClasses);
    
        Compute.Return(exp, sums, maxs);
    
        return x.Create(x.Value.CreateAlike(result), x.Gradient.Zeros(), Backward, [x]);
    
        void Backward(ITensor t)
        {
            var temp = Compute.Get(aidx, totalSize);
            var dot = Compute.Get(aidx, batchSize);
        
            Compute.Call(Compute.ElementwiseMultiplyKernels, temp, t.Gradient.Data, result);
            Compute.Call(BatchedSumRowKernels, dot, temp, numClasses);
            Compute.Call(BatchedSoftmaxBackwardKernels, x.Gradient.Data, t.Gradient.Data, result, dot, numClasses);
        
            Compute.Return(temp, dot);
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
