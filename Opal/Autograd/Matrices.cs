using Opal.Mathematics;

namespace Opal.Autograd;

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