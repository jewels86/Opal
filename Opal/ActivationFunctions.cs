using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

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
    
    #region Overloads
    public static Tensor<T> ReLu<T>(Tensor<T> x) where T : notnull => (Tensor<T>)ReLu(x as ITensor);
    public static Tensor<T> Sigmoid<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Sigmoid(x as ITensor);
    public static Tensor<T> Tanh<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Tanh(x as ITensor);
    public static Tensor<T> Identity<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Identity(x as ITensor);
    public static Tensor<T> Softmax<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Softmax(x as ITensor);
    #endregion
}
