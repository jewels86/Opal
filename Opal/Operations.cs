using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Jewels.Lazulite;
using Opal.NNs;

namespace Opal;

public static partial class Operations
{
    public static int DefaultAcceleratorIndex { get; }

    static Operations()
    {
        Compute.InitializeExtraKernels();
        DefaultAcceleratorIndex = Compute.RequestAccelerator();
    }

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> ElementwiseFloatMulAndSubKernels { get; }
        = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> r, float alpha) =>
            r[i] = b[i] - a[i] * alpha);

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseTripleAddKernels { get; } 
        = Compute.Load((i, a, b, c, r) => r[i] = a[i] + b[i] + c[i]);
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseLstmStateKernels { get; } 
        = Compute.Load((i, forget, state, input, cell, r) => r[i] = forget[i] * state[i] + input[i] * cell[i]);
    

    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(a.AcceleratorIndex, Compute.ElementwiseAddKernels, a.Value, b.Value, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            Compute.Call(a.AcceleratorIndex, Compute.ElementwiseAddKernels, t.Gradient.Data, a.Gradient.Data, a.Gradient.Data);
            Compute.Call(b.AcceleratorIndex, Compute.ElementwiseAddKernels, t.Gradient.Data, b.Gradient.Data, b.Gradient.Data);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
        }
    }

    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c, bool disposeA = true, bool disposeB = true, bool disposeC = true) where T : notnull
    {
        var result = Compute.GetLike(a.Value);
        Compute.Call(a.AcceleratorIndex, ElementwiseTripleAddKernels, a.Value, b.Value, c.Value, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b, c]);
        
        void Backward(ITensor t)
        {
            Compute.Call(a.AcceleratorIndex, Compute.ElementwiseAddKernels, t.Gradient.Data, a.Gradient.Data, a.Gradient.Data);
            Compute.Call(b.AcceleratorIndex, Compute.ElementwiseAddKernels, t.Gradient.Data, b.Gradient.Data, b.Gradient.Data);
            Compute.Call(c.AcceleratorIndex, Compute.ElementwiseAddKernels, t.Gradient.Data, c.Gradient.Data, c.Gradient.Data);
            if (disposeA) a.Dispose();
            if (disposeB) b.Dispose();
            if (disposeC) c.Dispose();
        }
    }

    public static Tensor<T> Concat<T>(Tensor<T> a, Tensor<T> b, bool disposeA = true, bool disposeB = true) where T : notnull
    {
        int aidx = a.AcceleratorIndex;
        var result = Compute.Get(a.Value.AcceleratorIndex, a.Value.TotalSize + b.Value.TotalSize);
        Compute.Call(aidx, Compute.ConcatKernels, a.Value.Data, b.Value.Data, result);
        return new(a.Value.Create(result, a.Value.Shape), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor t)
        {
            var slicedA = Compute.GetLike(a.Gradient);
            var slicedB = Compute.GetLike(b.Gradient);
            Compute.Call(aidx, Compute.SliceKernels, t.Gradient.Data, slicedA, 0, a.Value.TotalSize);
            Compute.Call(aidx, Compute.SliceKernels, t.Gradient.Data, slicedB, a.Value.TotalSize, b.Value.TotalSize);
            
            Compute.Call(aidx, Compute.ElementwiseAddKernels, t.Gradient.Data, a.Gradient, a.Gradient);
            Compute.Call(aidx, Compute.ElementwiseAddKernels, t.Gradient.Data, b.Gradient, b.Gradient);
            
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
            int aidx = t.Value.AcceleratorIndex;
            Compute.Call(aidx, ElementwiseTripleAddKernels, forget.Gradient.Data, state.Gradient.Data, t.Gradient.Data, forget.Gradient.Data);
            Compute.Call(aidx, ElementwiseTripleAddKernels, state.Gradient.Data, forget.Gradient.Data, t.Gradient.Data, state.Gradient.Data);
            Compute.Call(aidx, ElementwiseTripleAddKernels, input.Gradient.Data, cell.Gradient.Data, t.Gradient.Data, input.Gradient.Data);
            Compute.Call(aidx, ElementwiseTripleAddKernels, cell.Gradient.Data, input.Gradient.Data, t.Gradient.Data, cell.Gradient.Data);
            if (disposeForget) forget.Dispose();
            if (disposeState) state.Dispose();
            if (disposeInput) input.Dispose();
            if (disposeCell) cell.Dispose();
        }
    }

    public static Tensor<T> LstmHidden<T>(Tensor<T> output, Tensor<T> state, bool disposeOutput = true, bool disposeState = true) where T : notnull
    {
        var result = Compute.GetLike(output.Value);
        Compute.Call(output.AcceleratorIndex, Compute.ElementwiseMultiplyKernels, output.Value, state.Value, result);
        return new(state.Value.Create(result, state.Value.Shape), state.Value.Zeros(), Backward, [output, state]);
        
        void Backward(ITensor t)
        {
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
}