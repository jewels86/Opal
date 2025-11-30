using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static partial class Operations
{
    internal static Compute Compute => Compute.Instance;
    public static int DefaultAcceleratorIndex { get; set; } = -1;

    public static void Dispose() => Compute.ClearAll();

    public static void Sync() => Compute.Synchronize(DefaultAcceleratorIndex);
    
    #region Value Operations
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ElementwiseFloatMulAndSubKernels { get; }
        = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> r, float alpha) =>
            r[i] = b[i] - a[i] * alpha);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseTripleAddKernels { get; } 
        = Compute.Load((i, a, b, c, r) => r[i] = a[i] + b[i] + c[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseLstmStateKernels { get; } 
        = Compute.Load((i, forget, state, input, cell, r) => 
            r[i] = forget[i] * state[i] + input[i] * cell[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseAccumulateKernels { get; } = Compute.Load((i, a, r) => r[i] += a[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseNegAccumulateKernels { get; } = Compute.Load((i, a, r) => r[i] -= a[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseMulAccumulateKernels { get; } = Compute.Load((i, a, b, r) => r[i] += b[i] * a[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseMulScalarAccumulateKernels { get; } = Compute.Load((i, a, b, r) => r[i] += b[0] * a[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] ElementwiseDivAccumulateKernels { get; } = Compute.Load((i, a, b, r) => r[i] += a[i] / b[i]);
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseDivBackwardKernels { get; } 
        = Compute.Load((i, a, b, grad, r) => r[i] -= grad[i] * a[i] / (b[i] * b[i]));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] ElementwiseClipGradByNormKernels1 { get; } 
        = Compute.Load((i, grad, tn2) => Atomic.Add(ref tn2[0], grad[i] * grad[i]));
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>[] ElementwiseClipGradByNormKernels2 { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> grad, 
            ArrayView1D<float, Stride1D.Dense> tn,
            float maxNorm) => grad[i] = tn[0] > maxNorm ? grad[i] * maxNorm / tn[0] : grad[i]);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float>[] ElementwiseClampKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> grad, 
            float min, float max) => grad[i] = XMath.Max(min, XMath.Min(max, grad[i])));
    
    public static Value<T> Multiply<T>(Value<T> a, Value<T> b) where T : notnull => 
        a.Create(Compute.BinaryCall(Compute.ElementwiseMultiplyKernels, a.Data, b.Data), a.Shape);
    public static void Multiply(IValue a, IValue b, IValue result) => 
        Compute.Call(Compute.ElementwiseMultiplyKernels, a.Data, b.Data, result.Data);
    
    public static Value<T> Add<T>(Value<T> a, Value<T> b) where T : notnull => 
        a.Create(Compute.BinaryCall(Compute.ElementwiseAddKernels, a.Data, b.Data), a.Shape);
    public static void Add(IValue a, IValue b, IValue result) => 
        Compute.Call(Compute.ElementwiseAddKernels, a.Data, b.Data, result.Data);
    
    public static Value<T> Subtract<T>(Value<T> a, Value<T> b) where T : notnull => 
        a.Create(Compute.BinaryCall(Compute.ElementwiseSubtractKernels, a.Data, b.Data), a.Shape);
    public static void Subtract(IValue a, IValue b, IValue result) => 
        Compute.Call(Compute.ElementwiseSubtractKernels, a.Data, b.Data, result.Data);
    
    public static Value<T> Divide<T>(Value<T> a, Value<T> b) where T : notnull => 
        a.Create(Compute.BinaryCall(Compute.ElementwiseDivideKernels, a.Data, b.Data), a.Shape);
    public static void Divide<T>(Value<T> a, Value<T> b, Value<T> result) where T : notnull => 
        Compute.Call(Compute.ElementwiseDivideKernels, a.Data, b.Data, result.Data);
    
    public static void Accumulate(IValue a, IValue result) => 
        Compute.Call(ElementwiseAccumulateKernels, a.Data, result.Data);
    public static void NegAccumulate(IValue a, IValue result) => 
        Compute.Call(ElementwiseNegAccumulateKernels, a.Data, result.Data);
    public static void MulAccumulate(IValue a, IValue b, IValue result) => 
        Compute.Call(ElementwiseMulAccumulateKernels, a.Data, b.Data, result.Data);
    public static void MulScalarAccumulate(IValue a, IValue b, IValue result) => 
        Compute.Call(ElementwiseMulScalarAccumulateKernels, a.Data, b.Data, result.Data);
    public static void DivAccumulate(IValue a, IValue b, IValue result) => 
        Compute.Call(ElementwiseDivAccumulateKernels, a.Data, b.Data, result.Data);
    public static void DivBackward(IValue a, IValue b, IValue grad, IValue result) => 
        Compute.Call(ElementwiseDivBackwardKernels, a.Data, b.Data, grad.Data, result.Data);
    #endregion

    #region Tensor Operations
    #region Add & Subtract
    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(Compute.ElementwiseAddKernels, a.Value, b.Value, result);
        return new(a.Value.CreateAlike(result), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            Accumulate(t.Gradient, a.Gradient);
            Accumulate(t.Gradient, b.Gradient);
        }
    }

    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(ElementwiseTripleAddKernels, a.Value, b.Value, c.Value, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b, c]);
        
        void Backward(ITensor t)
        {
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, a.Gradient.Data, a.Gradient.Data);
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, b.Gradient.Data, b.Gradient.Data);
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, c.Gradient.Data, c.Gradient.Data);
        }
    }
    public static Tensor<T> Subtract<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Subtract(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            Accumulate(t.Gradient, a.Gradient);
            NegAccumulate(t.Gradient, b.Gradient);
        }
    }
    #endregion
    #region Multiply & Divide
    public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Multiply(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            MulAccumulate(t.Gradient, b.Value, a.Gradient);
            MulAccumulate(t.Gradient, a.Value, b.Gradient);
        }
    }

    public static Tensor<T> Divide<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Divide(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            DivAccumulate(t.Gradient, b.Value, a.Gradient);
            DivBackward(a.Value, b.Value, t.Gradient, b.Gradient);
        }
    }
    #endregion

    public static Tensor<T> Concat<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        var result = Compute.Get(a.Value.AcceleratorIndex, a.Value.TotalSize + b.Value.TotalSize);
        Compute.Call(Compute.ConcatKernels, a.Value.Data, b.Value.Data, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            var slicedA = Compute.GetLike(a.Gradient);
            var slicedB = Compute.GetLike(b.Gradient);
            Compute.Call(Compute.SliceKernels, t.Gradient.Data, slicedA, 0, a.Value.TotalSize);
            Compute.Call(Compute.SliceKernels, t.Gradient.Data, slicedB, a.Value.TotalSize, b.Value.TotalSize);
            
            Compute.Call(Compute.ElementwiseAddKernels, slicedA, a.Gradient, a.Gradient);
            Compute.Call(Compute.ElementwiseAddKernels, slicedB, b.Gradient, b.Gradient);
        }
    }
    
    #region LSTM
    public static Tensor<T> LstmState<T>(Tensor<T> forget, Tensor<T> state, Tensor<T> input, Tensor<T> cell) where T : notnull
    {
        var result = Compute.GetLike(state.Value);
        Compute.Call(
            state.AcceleratorIndex, ElementwiseLstmStateKernels, 
            forget.Value.Data.IntExtent, forget.Value.Data, 
            state.Value.Data, input.Value.Data, 
            cell.Value.Data, result);
        
        return new(state.Value.Create(result, state.Value.Shape), state.Value.Zeros(), Backward, [forget, state, input, cell]);

        void Backward(ITensor t)
        {
            MulAccumulate(state.Value, t.Gradient, forget.Gradient);
            MulAccumulate(forget.Value, t.Gradient, state.Gradient);
            MulAccumulate(input.Value, t.Gradient, cell.Gradient);
            MulAccumulate(cell.Value, t.Gradient, input.Gradient);
        }
    }
    #endregion
    #endregion
    
    #region Other stuff
    public static void ClipGradientsByNorm(float maxNorm, params ITensor[] tensors)
    {
        var (totalNorm2, totalNorm) = (NewValue(0), NewValue(0));

        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ElementwiseClipGradByNormKernels1, grad.Data, totalNorm2);
        
        Compute.Call(Compute.ElementwiseSqrtKernels, totalNorm2, totalNorm);
        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ElementwiseClipGradByNormKernels2, grad.Data, totalNorm, maxNorm);
    }

    public static void ClipGradientsByValue(float min, float max, params ITensor[] tensors)
    {
        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ElementwiseClampKernels, grad.Data, min, max);
    }
    #endregion
}