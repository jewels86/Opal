using Opal.Autograd;
using static Opal.Mathematics.Matrices;

namespace Opal.Mathematics;

public static class LossFunctions
{
    #region Scalars
    public static readonly LossFunction<double> MeanSquaredError = new(
        (predicted, actual) =>
        {
            var loss = Math.Pow(predicted.Value - actual, 2);
            return new Tensor<double>(loss, [predicted], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                predicted.Gradient += 2 * (predicted.Value - actual) * output.Gradient;
            }
        }
    );
    #endregion
    #region Vectors
    public static readonly LossFunction<double[]> MeanSquaredErrorVector = new(
        (predicted, actual) =>
        {
            if (predicted.Value.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            
            double sum = 0;
            for (int i = 0; i < predicted.Value.Length; i++)
                sum += Math.Pow(predicted.Value[i] - actual[i], 2);
            var loss = sum / predicted.Value.Length;
            
            return new Tensor<double>(loss, [predicted], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                double[] gradient = new double[predicted.Value.Length];
                for (int i = 0; i < predicted.Value.Length; i++)
                    gradient[i] = 2 * (predicted.Value[i] - actual[i]) / predicted.Value.Length * output.Gradient;
                predicted.Gradient = Vectors.Add(predicted.Gradient, gradient);
            }
        }
    );
    
    public static readonly LossFunction<double[]> CrossEntropy = new(
        (predicted, actual) =>
        {
            if (predicted.Value.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            
            double sum = 0;
            for (int i = 0; i < predicted.Value.Length; i++)
            {
                if (Math.Abs(actual[i] - 1) < 1e-15)
                    sum -= Math.Log(predicted.Value[i] + 1e-15);
                else
                    sum -= Math.Log(1 - predicted.Value[i] + 1e-15);
            }
            var loss = sum / predicted.Value.Length;
            
            return new Tensor<double>(loss, [predicted], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                double[] gradient = new double[predicted.Value.Length];
                for (int i = 0; i < predicted.Value.Length; i++)
                    gradient[i] = (predicted.Value[i] - actual[i]) / ((predicted.Value[i] * (1 - predicted.Value[i])) + 1e-15) / predicted.Value.Length * output.Gradient;
                predicted.Gradient = Vectors.Add(predicted.Gradient, gradient);
            }
        }
    );
    #endregion
    #region Matrices
    public static readonly LossFunction<double[,]> MeanSquaredErrorMatrix = new(
        (predicted, actual) =>
        {
            if (predicted.Value.GetLength(0) != actual.GetLength(0) || predicted.Value.GetLength(1) != actual.GetLength(1))
                throw new ArgumentException("Matrices must be of the same dimensions.");
            
            double sum = 0;
            int rows = predicted.Value.GetLength(0), cols = predicted.Value.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    sum += Math.Pow(predicted.Value[i, j] - actual[i, j], 2);
            var loss = sum / (rows * cols);
            
            return new Tensor<double>(loss, [predicted], Backwards, 0.0);
            
            void Backwards(Tensor<double> output)
            {
                double[,] gradient = new double[rows, cols];
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        gradient[i, j] = 2 * (predicted.Value[i, j] - actual[i, j]) / (rows * cols) * output.Gradient;
                predicted.Gradient = Add(predicted.Gradient, gradient);
            }
        }
    );
    #endregion
}

public record struct LossFunction<T>(Func<Tensor<T>, T, Tensor<double>> Function) where T : notnull;
