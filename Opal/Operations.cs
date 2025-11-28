using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Jewels.Lazulite;
using Opal.NNs;

namespace Opal;

public static partial class Operations
{
    internal static Compute Compute => Compute.Instance;
    
    public static int DefaultAcceleratorIndex { get; set; }

    static Operations()
    {
        Compute.InitializeExtraKernels();
        DefaultAcceleratorIndex = Compute.RequestAccelerator();
    }

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
    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(Compute.ElementwiseAddKernels, a.Value, b.Value, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            Accumulate(t.Gradient, a.Gradient);
            Accumulate(t.Gradient, b.Gradient);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }

    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c, bool disposeA = true, bool disposeB = true, bool disposeC = true) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(ElementwiseTripleAddKernels, a.Value, b.Value, c.Value, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b, c]);
        
        void Backward(ITensor t)
        {
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, a.Gradient.Data, a.Gradient.Data);
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, b.Gradient.Data, b.Gradient.Data);
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, c.Gradient.Data, c.Gradient.Data);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
            if (disposeC) c.Dispose();
        }
    }
    public static Tensor<T> Subtract<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        return new(Subtract(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            Accumulate(t.Gradient, a.Gradient);
            NegAccumulate(t.Gradient, b.Gradient);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }
    #endregion
    #region Multiply & Divide
    public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        return new(Multiply(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            MulAccumulate(t.Gradient, b.Value, a.Gradient);
            MulAccumulate(t.Gradient, a.Value, b.Gradient);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }

    public static Tensor<T> Divide<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        return new(Divide(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);;
        
        void Backward(ITensor t)
        {
            DivAccumulate(t.Gradient, b.Value, a.Gradient);
            DivBackward(a.Value, b.Value, t.Gradient, b.Gradient);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }
    #endregion

    public static Tensor<T> Concat<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        int aidx = a.AcceleratorIndex;
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
            
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }
    
    #region LSTM
    public static Tensor<T> LstmState<T>(Tensor<T> forget, Tensor<T> state, Tensor<T> input, Tensor<T> cell,
        bool disposeForget = true, bool disposeState = true, bool disposeInput = true, bool disposeCell = true) where T : notnull
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
            // these are REALLY wrong
            int aidx = t.Value.AcceleratorIndex;
            Compute.Call(ElementwiseTripleAddKernels, forget.Gradient.Data, state.Gradient.Data, t.Gradient.Data, forget.Gradient.Data);
            Compute.Call(ElementwiseTripleAddKernels, state.Gradient.Data, forget.Gradient.Data, t.Gradient.Data, state.Gradient.Data);
            Compute.Call(ElementwiseTripleAddKernels, input.Gradient.Data, cell.Gradient.Data, t.Gradient.Data, input.Gradient.Data);
            Compute.Call(ElementwiseTripleAddKernels, cell.Gradient.Data, input.Gradient.Data, t.Gradient.Data, cell.Gradient.Data);
            if (disposeForget) forget.Dispose();
            if (disposeState) state.Dispose(); 
            if (disposeInput) input.Dispose();
            if (disposeCell) cell.Dispose();
        }
    }

    public static Tensor<T> LstmHidden<T>(Tensor<T> output, Tensor<T> state, bool disposeOutput = true, bool disposeState = true) where T : notnull
    {
        var result = Compute.GetLike(output.Value);
        Compute.Call(Compute.ElementwiseMultiplyKernels, output.Value, state.Value, result);
        return new(state.Value.Create(result, state.Value.Shape), state.Value.Zeros(), Backward, [output, state]);
        
        void Backward(ITensor t)
        {
            // these are REALLY wrong
            Compute.BinaryCallChain(t.Gradient.Data, output.Gradient.Data, 
                (Compute.ElementwiseMultiplyKernels, state.Gradient.Data), 
                (Compute.ElementwiseAddKernels, output.Gradient.Data));
            Compute.BinaryCallChain(t.Gradient.Data, state.Gradient.Data, 
                (Compute.ElementwiseMultiplyKernels, output.Gradient.Data), 
                (Compute.ElementwiseAddKernels, state.Gradient.Data));
            if (disposeOutput) output.Dispose();
            if (disposeState) state.Dispose();
        }
    }
    #endregion
    #endregion
}