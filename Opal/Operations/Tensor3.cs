using ILGPU;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal;

public partial class Operations
{
    public static Tensor<float[,,]> New(float[,,] tensor3, float[,,]? gradient = null, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null, int? aidx = null) => new(
        new TensorValue3(tensor3, aidx ?? DefaultAcceleratorIndex),
        new TensorValue3(gradient ?? Fill(0, tensor3.GetLength(0), tensor3.GetLength(1), tensor3.GetLength(2)), 
            aidx ?? DefaultAcceleratorIndex), backwardAction, inputs);
    public static Tensor<float[,,]> New(Value<float[,,]> tensor3, Value<float[,,]> gradient, Action<ITensor>? backwardAction = null, List<ITensor>? inputs = null) => 
        new(tensor3, gradient, backwardAction, inputs);

    
    #region Kernels
    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int>[]
        AddVectorToTensor3Kernel { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> tensor,
        ArrayView1D<float, Stride1D.Dense> vector,
        ArrayView1D<float, Stride1D.Dense> result, int n) => result[i] = tensor[i] + vector[i % n]);

    public static Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int>[] AddVectorToTensor3BackwardKernel { get; }
        = Compute.Load((Index1D featureIdx,
            ArrayView1D<float, Stride1D.Dense> vectorGrad,
            ArrayView1D<float, Stride1D.Dense> grad, int features, int seqLength, int batchSize) =>
        {
            float sum = 0;
            for (int batch = 0; batch < batchSize; batch++)
            for (int seq = 0; seq < seqLength; seq++)
                sum += grad[(batch * seqLength + seq) * features + featureIdx];
            vectorGrad[featureIdx] += sum;
        });
    #endregion
    
    public static Tensor<float[,,]> Add(Tensor<float[,,]> tensor, Tensor<float[]> vector)
    {
        var (aidx, batchSize, seqLength, features) = (tensor.AcceleratorIndex, tensor.Value.Shape[0], tensor.Value.Shape[1], tensor.Value.Shape[2]);
        var result = new TensorValue3(Compute.Get(aidx, batchSize * seqLength * features), [batchSize, seqLength, features]);
        Compute.Call(AddVectorToTensor3Kernel, tensor.Value.Data, vector.Value.Data, result.Data, features);

        return new(result, result.Zeros(), Backward, [tensor, vector]);

        void Backward(ITensor t)
        {
            Compute.Call(Compute.ElementwiseAddKernels, t.Gradient.Data, t.Gradient.Data, t.Gradient.Data);
            Compute.Call(AddVectorToTensor3BackwardKernel, vector.Gradient.Data, t.Gradient.Data, features, seqLength, batchSize);
        }
    }

    public static Tensor<float[,,]> BatchedMatrixMultiply(Tensor<float[,,]> a, Tensor<float[,]> b, bool transposeB = false)
    {
        var (aidx, batchSize, seqLength, aFeatures) = (a.AcceleratorIndex, a.Value.Shape[0], a.Value.Shape[1], a.Value.Shape[2]);
        var (b0, b1) = (b.Value.Shape[0], b.Value.Shape[1]);
        var n = transposeB ? b0 : b1;
        
        var result = new TensorValue3(Compute.Get(aidx, batchSize * seqLength * n), [batchSize, seqLength, n]);
        
        // For each timestep, do batched matmul
        for (int t = 0; t < seqLength; t++)
        {
            var inputOffset = t * batchSize * aFeatures;
            var outputOffset = t * batchSize * n;
            
            // [batch, aFeatures] * [b0, b1] = [batch, n]
            Compute.MatrixMultiply(
                a.Value.Data.View.SubView(inputOffset, batchSize * aFeatures),
                b.Value.Data,
                result.Data.View.SubView(outputOffset, batchSize * n),
                batchSize, aFeatures, b0, b1,
                transposeA: false, transposeB: transposeB
            );
        }
        
        return new(result, result.Zeros(), Backward, [a, b]);

        void Backward(ITensor tensor)
        {
            var gradA = Compute.Get(aidx, batchSize * seqLength * aFeatures);
            var gradB = Compute.Get(aidx, b0 * b1);
            
            for (int t = 0; t < seqLength; t++)
            {
                var inputOffset = t * batchSize * aFeatures;
                var outputOffset = t * batchSize * n;
                
                var tempGradA = Compute.Get(aidx, batchSize * aFeatures);
                var tempGradB = Compute.Get(aidx, b0 * b1);
                
                if (!transposeB)
                {
                    // d/da: grad * b^T
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data.View.SubView(outputOffset, batchSize * n),
                        b.Value.Data,
                        tempGradA,
                        batchSize, n, b0, b1,
                        transposeA: false, transposeB: true
                    );
                    
                    // d/db: a^T * grad
                    Compute.MatrixMultiply(
                        a.Value.Data.View.SubView(inputOffset, batchSize * aFeatures),
                        tensor.Gradient.Data.View.SubView(outputOffset, batchSize * n),
                        tempGradB,
                        batchSize, aFeatures, batchSize, n,
                        transposeA: true, transposeB: false
                    );
                }
                else
                {
                    // d/da: grad * b
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data.View.SubView(outputOffset, batchSize * n),
                        b.Value.Data,
                        tempGradA,
                        batchSize, n, b0, b1,
                        transposeA: false, transposeB: false
                    );
                    
                    // d/db: grad^T * a
                    Compute.MatrixMultiply(
                        tensor.Gradient.Data.View.SubView(outputOffset, batchSize * n),
                        a.Value.Data.View.SubView(inputOffset, batchSize * aFeatures),
                        tempGradB,
                        batchSize, n, batchSize, aFeatures,
                        transposeA: true, transposeB: false
                    );
                }
                
                Compute.Call(ElementwiseAccumulateKernels, tempGradA, gradA.View.SubView(inputOffset, batchSize * aFeatures));
                Compute.Call(ElementwiseAccumulateKernels, tempGradB, gradB);
                Compute.Return(tempGradA, tempGradB);
            }
            
            Compute.Call(ElementwiseAccumulateKernels, gradA, a.Gradient);
            Compute.Call(ElementwiseAccumulateKernels, gradB, b.Gradient);
            Compute.Return(gradA, gradB);
        }
    }
}