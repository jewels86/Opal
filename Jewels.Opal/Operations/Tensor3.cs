using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Jewels.Opal;

public partial class Operations
{
    public static Tensor<float[,,]> New(float[,,] tensor3, float[,,]? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null, int? aidx = null) => new(
        new TensorValue3(tensor3, aidx ?? DefaultAcceleratorIndex),
        new TensorValue3(gradient ?? Fill(0, tensor3.GetLength(0), tensor3.GetLength(1), tensor3.GetLength(2)), 
            aidx ?? DefaultAcceleratorIndex), backwardAction, inputs);
    public static Tensor<float[,,]> New(Value<float[,,]> tensor3, Value<float[,,]>? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new(tensor3, gradient ?? tensor3.Zeros(), backwardAction, inputs);

    
    #region Kernels
    /// <summary>
    /// (r, tensor3, timestep, seqLen, features) => r[i] = tensor3[batch, timestep, feature]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, int>[]
        GetSliceKernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> tensor3,
        int timestep, int seqLen, int features) => 
        {
            int batch = i / features;
            int feature = i % features;
            result[i] = tensor3[batch * seqLen * features + timestep * features + feature];
        });
    
    /// <summary>
    /// (result, matrix, timestep seqLen, features) => result[batch, timestep, feature] = matrix[i]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, int>[]
        SetSliceKernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> matrix,
        int timestep, int seqLen, int features) => 
        {
            int batch = i / features;
            int feature = i % features;
            result[batch * seqLen * features + timestep * features + feature] = matrix[i];
        });
    
    /// <summary>
    /// (sliceGrad, tensor3Grad, timestep, seqLen, features) => tensor3Grad[batch, timestep, feature] += sliceGrad[i]
    /// </summary>
    public static Action<Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, int, int, int>[]
        GetSliceBackwardKernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> sliceGrad,
        ArrayView1D<float, Stride1D.Dense> tensor3Grad,
        int timestep, int seqLen, int features) => 
        {
            int batch = i / features;
            int feature = i % features;
            Atomic.Add(ref tensor3Grad[batch * seqLen * features + timestep * features + feature], sliceGrad[i]);
        });
    
    #endregion
    
    public static Tensor<float[,]> GetSlice(Tensor<float[,,]> tensor3, int timestep)
    {
        var (aidx, batch, seqLen, features) = (tensor3.AcceleratorIndex, 
            tensor3.Value.Shape[0], tensor3.Value.Shape[1], tensor3.Value.Shape[2]);
        var result = new MatrixValue(Compute.Get(aidx, batch * features), [batch, features]);
        Compute.Call(GetSliceKernel, result.Data, tensor3.Value, timestep, seqLen, features);

        return new(result, result.Zeros(), Backward, [tensor3]);

        void Backward(ITensor tensor) => Compute.Call(GetSliceBackwardKernel, tensor.Gradient.Data, tensor3.Gradient.Data, timestep, seqLen, features);
    }

    public static Tensor<float[,,]> SetSlice(Tensor<float[,,]> tensor3, Tensor<float[,]> slice, int timestep)
    {
        var (aidx, batch, seqLen, features) = (tensor3.AcceleratorIndex, tensor3.Value.Shape[0], tensor3.Value.Shape[1], tensor3.Value.Shape[2]);
        
        var result = new TensorValue3(Compute.Get(aidx, batch * seqLen * features), [batch, seqLen, features]);
        Compute.Copy(result.Data, tensor3.Value);
        Compute.Call(SetSliceKernel, tensor3.Value, slice.Value, timestep, seqLen, features);
        
        return new(result, result.Zeros(), Backward, [tensor3, slice]);

        void Backward(ITensor tensor)
        {
            AccumulateX(tensor3.Gradient, tensor.Gradient);
            var sliceGrad = Compute.Get(aidx, batch * features);
            Compute.Call(GetSliceKernel, tensor.Gradient.Data, sliceGrad, timestep, seqLen, features);
            Compute.Call(AccumulateKernels, slice.Gradient.Data, sliceGrad);
            Compute.Return(sliceGrad);
        }
    }
}