using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public static class LossFunctions
{
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>>> MeanSquaredErrorKernels { get; } = Compute.Load((i, x, t, r) 
        => r[i] = (x[i] - t[i]) * (x[i] - t[i]));

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> MeanSquaredErrorBackwardKernels { get; } 
        = Compute.Load((Index1D i, ArrayView1D<float, Stride1D.Dense> x, ArrayView1D<float, Stride1D.Dense> t, ArrayView1D<float, Stride1D.Dense> r, float n) => r[i] += 2 * (x[i] - t[i]) / n);
    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CrossEntropyKernels { get; } 
        = Compute.Load((i, pred, target, r) => r[i] = -target[i] * XMath.Log(pred[i]));

    public static List<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> CrossEntropyBackwardKernels { get; } 
        = Compute.Load((Index1D i, 
            ArrayView1D<float, Stride1D.Dense> pred, ArrayView1D<float, Stride1D.Dense> target, 
            ArrayView1D<float, Stride1D.Dense> grad, float n) => grad[i] += -target[i] / pred[i] / n);
    
    #region Vectors
    public static Tensor<float> MeanSquaredError(Tensor<float[]> predicted, VectorValue actual)
    {
        if (predicted.Value.TotalSize != actual.TotalSize)
            throw new ArgumentException("Vectors must be of the same length.");
        
        var aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, predicted.Value.TotalSize);
        var (temp1, temp2) = (Compute.Get(aidx, predicted.Value.TotalSize), Compute.Get(aidx, 2));
        Compute.Call(predicted.Value.AcceleratorIndex, MeanSquaredErrorKernels, predicted.Value.Data, actual.Data, temp1);
        Compute.Sum(temp1, temp2);
        Compute.Call(aidx, Compute.ElementwiseFloatMultiplyKernels, temp2, result, 1 / (float)actual.TotalSize);
        Compute.Return(temp1, temp2);
        
        return TensorOperations.New(new ScalarValue(result), new ScalarValue(0.0f, aidx), Backward, [predicted]);
        
        void Backward(ITensor t) => Compute.Call(aidx, MeanSquaredErrorBackwardKernels, predicted.Value.Data, actual.Data, predicted.Gradient.Data, actual.TotalSize);
    }
    
    public static Tensor<float> MeanSquaredError(Tensor<float[]> predicted, float[] actual)
    {
        if (predicted.Value.TotalSize != actual.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        
        return MeanSquaredError(predicted, new VectorValue(actual, predicted.Value.AcceleratorIndex));
    }
    
    public static Tensor<float> CrossEntropy(Tensor<float[]> predicted, Value<float[]> actual)
    {
        int aidx = predicted.Value.AcceleratorIndex;
        var result = Compute.Get(aidx, 1);
        var (temp1, temp2) = (Compute.Get(aidx, predicted.Value.TotalSize), Compute.Get(aidx, 1));
    
        Compute.Call(aidx, CrossEntropyKernels, predicted.Value.Data, actual, temp1);
        Compute.Sum(temp1, temp2);
        Compute.Call(aidx, Compute.ElementwiseFloatMultiplyKernels, temp2, result,  1 / (float)actual.TotalSize);
    
        Compute.Return(temp1, temp2);
    
        return TensorOperations.New(new ScalarValue(result), new ScalarValue(0.0f, aidx), Backward, [predicted]);
    
        void Backward(ITensor t) => Compute.Call(aidx, CrossEntropyBackwardKernels, 
            predicted.Value.Data, actual, predicted.Gradient.Data, actual.TotalSize);
    }

    
    public static Tensor<float> CrossEntropy(Tensor<float[]> predicted, float[] actual)
    {
        if (predicted.Value.TotalSize != actual.Length)
            throw new ArgumentException("Vectors must be of the same length.");
        
        return CrossEntropy(predicted, new VectorValue(actual, predicted.Value.AcceleratorIndex));
    }
    #endregion
}

