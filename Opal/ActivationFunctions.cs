using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static class ActivationFunctions
{
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ReLuBackwardKernels { get; }
        = Compute.Load((i, x, grad, r) => r[i] += x[i] > 0 ? grad[i] : 0.0f);
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SigmoidKernels { get; } 
        = Compute.Load((i, x, r) => r[i] = 1 / (1 + XMath.Exp(-x[i])));
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SigmoidBackwardKernels { get; } 
        = Compute.Load((i, x, grad, r) => r[i] += grad[i] * x[i] * (1 - x[i]));
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> TanhBackwardKernels { get; } 
        = Compute.Load((i, x, grad, r) => r[i] += grad[i] * (1 - x[i] * x[i]));
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AccumulateGradientKernels { get; } 
        = Compute.Load((i, grad, r) => r[i] += grad[i]);

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> SoftmaxBackwardKernels { get; }
        = Compute.Load((i, grad, softmax, dot, r) => r[i] += softmax[i] * (grad[i] - dot[0]));

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
            (a, r) => Compute.Call(a, Compute.ElementwiseFloatMaxKernels, x.Value.Data, r, 0.0f), 
            (a, _) => t => Compute.Call(a, ReLuBackwardKernels, x.Value.Data, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Sigmoid(ITensor x) => ActivationFunction(x, 
        (a, r) => Compute.Call(a, SigmoidKernels, x.Value.Data, r),
            (a, r) => t => Compute.Call(a, SigmoidBackwardKernels, r, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Tanh(ITensor x) =>
        ActivationFunction(x, (a, r) => Compute.Call(a, Compute.ElementwiseTanhKernels, x.Value.Data, r),
            (a, r) => t => Compute.Call(a, TanhBackwardKernels, r, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Identity(ITensor x) =>
        ActivationFunction(x, (a, r) => Compute.Call(a, Compute.CopyKernels, x.Value.Data, r), 
            (a, r) => t => Compute.Call(a, AccumulateGradientKernels, t.Gradient.Data, x.Gradient.Data));

    public static ITensor Softmax(ITensor x) =>
        ActivationFunction(x, (a, r) =>
        {
            var temps = Compute.Get(a, 2, x.Value.TotalSize);
            Compute.Call(a, Compute.ElementwiseExpKernels, x.Value.Data, temps[0]);
            Compute.Sum(temps[0], temps[1]);
            Compute.Call(a, Compute.ElementwiseScalarDivideKernels, temps[0], temps[1], r);
            Compute.Return(temps);
        }, (a, r) => t =>
        {
            var temp = Compute.Get(a, x.Value.TotalSize);
            Compute.Dot(t.Gradient.Data, r, temp);
            Compute.Call(a, SoftmaxBackwardKernels, t.Gradient.Data, r, temp, x.Gradient.Data);
            Compute.Return(temp);
        });
    
    #region Overloads
    public static Tensor<T> ReLu<T>(Tensor<T> x) where T : notnull => (Tensor<T>)ReLu(x as ITensor);
    public static Tensor<T> Sigmoid<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Sigmoid(x as ITensor);
    public static Tensor<T> Tanh<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Tanh(x as ITensor);
    public static Tensor<T> Identity<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Identity(x as ITensor);
    public static Tensor<T> Softmax<T>(Tensor<T> x) where T : notnull => (Tensor<T>)Softmax(x as ITensor);
    #endregion
}
