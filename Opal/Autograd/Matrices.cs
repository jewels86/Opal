using ILGPU;
using ILGPU.Runtime;
using Opal.Mathematics;

namespace Opal.Autograd;

public class MatrixTensor : Tensor<ITensorStorage<double[,]>>
{
    public MatrixTensor(
        ITensorStorage<double[,]> storage,
        List<object>? inputs,
        Action<Tensor<ITensorStorage<double[,]>>>? backward,
        ITensorStorage<double[,]> gradient)
        : base(storage, inputs, backward ?? (_ => { }), gradient) { }

    public static ITensorStorage<double[,]> CpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        return new CpuStorage<double[,]>
        {
            Data = matrix,
            Shape = [rows, cols],
            TotalElements = rows * cols
        };
    }

    public static ITensorStorage<double[,]> GpuMatrixStorage(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var buffer = Operations.Accelerator.Allocate2DDenseX<double>(new Index2D(rows, cols));
        buffer.CopyFromCPU(matrix);
        return new GpuMatrixStorage(buffer);
    }
}

public static partial class Operations
{
    public static Tensor<double[]> Multiply(Tensor<double[,]> matrix, Tensor<double[]> vector)
    {
        var result = Matrices.Multiply(matrix.Value, vector.Value);
        List<object> inputs = [matrix, vector];
        return new(result, inputs, Backwards, Vectors.Zeros(result.Length));

        void Backwards(Tensor<double[]> output)
        {
            var gradVector = Matrices.Multiply(Matrices.Transpose(matrix.Value), output.Gradient);
            vector.Gradient = Vectors.Add(vector.Gradient, gradVector);
            
            var gradMatrix = Matrices.OuterProduct(output.Gradient, vector.Value);
            matrix.Gradient = Matrices.Add(matrix.Gradient, gradMatrix);
        }
    }
    
    public static Tensor<double[]> Multiply(Tensor<double[]> input, Tensor<double[]>[] weights)
    {
        var dots = weights.Select(w => Dot(w, input)).ToArray();
        return VectorFromScalars(dots);
    }
}