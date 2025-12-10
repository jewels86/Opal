using System.Text;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

public static partial class Operations
{
    public static int DefaultAcceleratorIndex { get; set; } = -1;

    public static void Dispose() => Compute.ClearAll();

    public static void Sync() => Compute.Synchronize(DefaultAcceleratorIndex);
    
    #region Value Operations
    /// <summary>
    /// (r, a, b, alpha) => r = a - b * alpha
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, float>[] FloatMulAndSubKernels { get; } = Compute.Load((
            Index1D i, 
            ArrayView1D<float, Stride1D.Dense> r, 
            ArrayView1D<float, Stride1D.Dense> a, 
            ArrayView1D<float, Stride1D.Dense> b, float alpha) =>
            r[i] = a[i] - b[i] * alpha);
    
    /// <summary>
    /// (r, a, b, c) => r = a + b + c
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] TripleAddKernels { get; } = Compute.Load(
        (i, r, a, b, c) => r[i] = a[i] + b[i] + c[i]);
    
    /// <summary>
    /// (r, forget, state, input, cell) => r = forget * state + input * cell
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] LstmStateKernels { get; } = Compute.Load(
        (i, r, forget, state, input, cell) => r[i] = forget[i] * state[i] + input[i] * cell[i]);
    
    /// <summary>
    /// (r, a) => r += a
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] AccumulateKernels { get; } = Compute.Load(
        (i, r, a) => r[i] += a[i]);
    
    /// <summary>
    /// (r, a) => r -= a
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] DiminishKernels { get; } = Compute.Load(
        (i, r, a) => r[i] -= a[i]);
    
    /// <summary>
    /// (r, a, b) => r += a * b
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] MulAccumulateKernels { get; } = Compute.Load(
        (i, r, a, b) => r[i] += a[i] * b[i]);
    
    /// <summary>
    /// (r, a, b) => r[i] = a[i] * b[0]
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] MulScalarAccumulateKernels { get; } = Compute.Load(
        (i, r, a, b) => r[i] += a[i] * b[0]);
    
    /// <summary>
    /// (r, a, b) => r += a / b
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>[] DivAccumulateKernels { get; } = Compute.Load(
        (i, r, a, b) => r[i] += a[i] / b[i]);
    
    /// <summary>
    /// (r, a, b, grad) => r -= grad * a / b^2
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] DivBackwardKernels { get; } 
        = Compute.Load((i, r, a, b, grad) => r[i] -= grad[i] * a[i] / (b[i] * b[i]));
    
    /// <summary>
    /// (tn2, grad) => tn2[0] += grad[i] * grad[i] (atomic)
    /// </summary>
    public static Action<Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>[] ClipGradByNormKernels1 { get; } = Compute.Load(
        (i, tn2, grad) => Atomic.Add(ref tn2[0], grad[i] * grad[i]));
    
    /// <summary>
    /// (grad, tn, maxNorm) => grad[i] = (tn[0] > maxNorm and tn[0] > 0) ? grad[i] * maxNorm / tn[0] : grad[i]
    /// </summary>
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, float>[] ClipGradByNormKernels2 { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> grad,
        ArrayView1D<float, Stride1D.Dense> tn,
        float maxNorm) => grad[i] = (tn[0] > maxNorm && tn[0] > 0) ? grad[i] * maxNorm / tn[0] : grad[i]);

    /// <summary>
    /// (val, min, max) => val = Max(min, Min(max, val))
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float, float>[] ClampKernels { get; } = Compute.Load((Index1D i, 
        ArrayView1D<float, Stride1D.Dense> val,
        float min, float max) => val[i] = XMath.Max(min, XMath.Min(max, val[i])));
    
    /// <summary>
    /// (r, x) => r = -sin(x)
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] NegSinKernelsF { get; } = Compute.Load(
        (i, r, x) => r[i] = -XMath.Sin(x[i]));

    /// <summary>
    /// (r, x) => r = sec2(x) = 1 / cos2(x)
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] Sec2KernelsG { get; } = Compute.Load(
        (i, r, x) =>
    {
        var cosx = XMath.Cos(x[i]);
        r[i] = 1 / (cosx * cosx);
    });
    
    /// <summary>
    /// (r, x, grad) => r += grad/2x (x should be sqrt(original))
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] SqrtBackwardsKernelsY { get; } = Compute.Load(
        (i, r, x, grad) => r[i] += (0.5f * grad[i]) / x[i]);

    /// <summary>
    /// (r, x) => r = tanh(x)
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] TanhKernels { get; } = Compute.Load(
        (i, r, x) => r[i] = XMath.Tanh(x[i]));
    
    /// <summary>
    /// (r, x, grad) => r += grad * (1 - x^2)
    /// </summary>
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>[] TanhBackwardKernels { get; } 
        = Compute.Load((i, r, x, grad) => r[i] += grad[i] * (1 - (x[i] * x[i])));

    
    public static Value<T> Encase<T>(Value<T> alike, Action<Value<T>> compute) where T : notnull
    {
        var result = alike.Zeros();
        compute(result);
        return result;
    }
    
    public static Value<T> ElementwiseMultiply<T>(Value<T> a, Value<T> b) where T : notnull => Compute.ElementwiseMultiply(a, b);
    public static Value<T> Add<T>(Value<T> a, Value<T> b) where T : notnull => Compute.Add(a, b);
    public static Value<T> Subtract<T>(Value<T> a, Value<T> b) where T : notnull => Compute.Subtract(a, b);
    public static Value<T> Divide<T>(Value<T> a, Value<T> b) where T : notnull => Compute.Divide(a, b);

    public static Value<T> TripleAdd<T>(Value<T> a, Value<T> b, Value<T> c) where T : notnull => Encase(a, v => Compute.Call(TripleAddKernels, v, a, b, c));
    public static Value<T> Negate<T>(Value<T> x) where T : notnull => Encase(x, v => Compute.Call(Compute.NegateKernels, v, x));
    public static Value<T> Sqrt<T>(Value<T> x) where T : notnull => Encase(x, v => Compute.Sqrt(v, x));
    
    public static void AccumulateX(IValue result, IValue a) => Compute.Call(AccumulateKernels, result.Data, a.Data);
    public static void DiminishX(IValue result, IValue a) => Compute.Call(DiminishKernels, result.Data, a.Data);
    public static void MulAccumulateX(IValue result, IValue a, IValue b) => Compute.Call(MulAccumulateKernels, result.Data, a.Data, b.Data);
    public static void MulScalarAccumulateX(IValue result, IValue a, IValue b) => Compute.Call(MulScalarAccumulateKernels, result.Data, a.Data, b.Data);
    public static void DivAccumulateX(IValue result, IValue a, IValue b) => Compute.Call(DivAccumulateKernels, result.Data, a.Data, b.Data);
    public static void DivBackwardX(IValue result, IValue a, IValue b, IValue grad) => Compute.Call(DivBackwardKernels, result.Data, a.Data, b.Data, grad.Data);

    public static Value<T> Sine<T>(Value<T> x) where T : notnull => Compute.Sine(x);
    public static Value<T> Cosine<T>(Value<T> x) where T : notnull => Compute.Cosine(x);
    public static Value<T> Tangent<T>(Value<T> x) where T : notnull => Compute.Tangent(x);
    public static Value<T> Tanh<T>(Value<T> x) where T : notnull => Encase(x, v => Compute.Call(TanhKernels, v, x));
    #endregion

    #region Tensor Operations
    #region Add & Subtract
    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Add(a.Value, b.Value), a.Gradient.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            AccumulateX(a.Gradient, tensor.Gradient);
            AccumulateX(b.Gradient, tensor.Gradient);
        }
    }

    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c) where T : notnull
    {
        return new(TripleAdd<T>(a, b, c), a.Gradient.Zeros(), Backward, [a, b, c]);
        
        void Backward(ITensor tensor)
        {
            AccumulateX(a.Gradient, tensor.Gradient);
            AccumulateX(b.Gradient, tensor.Gradient);
            AccumulateX(c.Gradient, tensor.Gradient);
        }
    }
    public static Tensor<T> Subtract<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Subtract(a.Value, b.Value), a.Gradient.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            AccumulateX(a.Gradient, tensor.Gradient);
            DiminishX(b.Gradient, tensor.Gradient);
        }
    }
    #endregion
    #region Multiply & Divide
    public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(ElementwiseMultiply(a.Value, b.Value), a.Gradient.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            MulAccumulateX(a.Gradient, tensor.Gradient, b.Value);
            MulAccumulateX(b.Gradient, tensor.Gradient, a.Value);
        }
    }

    public static Tensor<T> Divide<T>(Tensor<T> a, Tensor<T> b) where T : notnull
    {
        return new(Divide(a.Value, b.Value), a.Value.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            DivAccumulateX(a.Gradient, tensor.Gradient, b.Value);
            DivBackwardX(b.Gradient, a.Value, b.Value, tensor.Gradient);
        }
    }
    #endregion

    public static Tensor<T> Negate<T>(Tensor<T> a) where T : notnull
    {
        return new(Negate(a.Value), a.Gradient.Zeros(), Backward, [a]);

        void Backward(ITensor tensor) => DiminishX(a.Gradient, tensor.Gradient);
    }

    public static Tensor<float[]> Concat(Tensor<float[]> a, Tensor<float[]> b)
    {
        var result = new VectorValue(new float[a.Value.TotalSize + b.Value.TotalSize], a.AcceleratorIndex);
        Compute.Call(Compute.ConcatKernels, result, a.Value, b.Value);
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var slicedA = Compute.GetLike(a.Gradient);
            var slicedB = Compute.GetLike(b.Gradient);
            Compute.Call(Compute.SliceKernels, slicedA, tensor.Gradient.Data, 0, a.Value.TotalSize);
            Compute.Call(Compute.SliceKernels, slicedB, tensor.Gradient.Data, a.Value.TotalSize, b.Value.TotalSize);
            
            Compute.Call(AccumulateKernels, a.Gradient, slicedA);
            Compute.Call(AccumulateKernels, b.Gradient, slicedB);
        }
    }

    public static Tensor<T> Square<T>(Tensor<T> a) where T : notnull => Multiply(a, a);

    public static Tensor<T> Sqrt<T>(Tensor<T> a) where T : notnull
    {
        var result = Sqrt(a.Value);
        return new(result, result.Zeros(), Backwards, [a]);

        void Backwards(ITensor t) => Compute.Call(SqrtBackwardsKernelsY, a.Gradient, result, t.Gradient.Data);
    }
    #region Trig
    public static Tensor<T> Sine<T>(Tensor<T> a) where T : notnull
    {
        return new(Sine(a.Value), a.Gradient.Zeros(), Backward, [a]);

        void Backward(ITensor t)
        {
            var cosx = Compute.Cosine(a.Value);
            MulAccumulateX(a.Gradient, t.Gradient, cosx);
        }
    }
    public static Tensor<T> Cosine<T>(Tensor<T> a) where T : notnull
    {
        return new(Cosine(a.Value), a.Gradient.Zeros(), Backward, [a]);

        void Backward(ITensor t)
        {
            var grad = Compute.GetLike(a.Gradient);
            Compute.Call(NegSinKernelsF, grad, a.Value);
            Compute.Call(MulAccumulateKernels, a.Gradient, grad, t.Gradient.Data);
        }
    }
    
    public static Tensor<T> Tangent<T>(Tensor<T> a) where T : notnull
    {
        return new(Tangent(a.Value), a.Gradient.Zeros(), Backward, [a]);

        void Backward(ITensor t)
        {
            var grad = Compute.GetLike(a.Gradient);
            Compute.Call(Sec2KernelsG, grad, a.Value);
            Compute.Call(MulAccumulateKernels, a.Gradient, grad, t.Gradient.Data);
        }
    }

    public static Tensor<T> Tanh<T>(Tensor<T> val) where T : notnull
    {
        var result = Tanh(val.Value);
        return new(result, result.Zeros(),Backward, [val]);

        void Backward(ITensor tensor) => Compute.Call(TanhBackwardKernels, val.Gradient, result, tensor.Gradient.Data);
    }
    #endregion
    
    #region LSTM
    public static Tensor<T> LstmState<T>(Tensor<T> forget, Tensor<T> state, Tensor<T> input, Tensor<T> cell) where T : notnull
    {
        var result = Compute.GetLike(state.Value);
        Compute.Call(LstmStateKernels,  result, forget.Value, state.Value, input.Value, cell.Value);
        return new(state.Value.CreateAlike(result), state.Value.Zeros(), Backward, [forget, state, input, cell]);

        void Backward(ITensor t)
        {
            MulAccumulateX(forget.Gradient, state.Value, t.Gradient);
            MulAccumulateX(state.Gradient, forget.Value, t.Gradient);
            MulAccumulateX(cell.Gradient, input.Value, t.Gradient);
            MulAccumulateX(input.Gradient, cell.Value, t.Gradient);
        }
    }
    #endregion
    #endregion
    
    #region Other stuff
    public static void ClipGradientsByNorm(float maxNorm, params ITensor[] tensors)
    {
        var (totalNorm2, totalNorm) = (NewValue(0), NewValue(0));

        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ClipGradByNormKernels1, totalNorm2, grad.Data);
        
        Compute.Sqrt(totalNorm, totalNorm2);
        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ClipGradByNormKernels2, grad.Data, totalNorm, maxNorm); // check this
    }

    public static void ClipGradientsByValue(float min, float max, params ITensor[] tensors)
    {
        foreach (var grad in tensors.Select(t => t.Gradient)) 
            Compute.Call(ClampKernels, grad.Data, min, max);
    }
    #endregion
    
    public static string ToString(float x) => x.ToString("0.00");
    
    public static string ToString(float[] vector) => "[" + string.Join(", ", vector.Select(ToString)) + "]";
    public static string ToString(int[] vector) => "[" +string.Join(", ", vector) + "]";

    public static string ToString(float[,] matrix)
    {
        StringBuilder sb = new("[");
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            sb.Append(" [");
            sb.Append(ToString(matrix[i, 0]));
            for (int j = 1; j < matrix.GetLength(1); j++) sb.Append($", {ToString(matrix[i, j])}");
            sb.Append("] ");
        }
        return sb.Append(']').ToString();
    }
}