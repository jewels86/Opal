namespace Opal.Mathematics;

public static class LossFunctions
{
    #region Scalars
    public static readonly LossFunction<double> MeanSquaredError = new(
        (predicted, actual) => Math.Pow(predicted - actual, 2),
        (predicted, actual) => 2 * (predicted - actual)
    );
    #endregion
    #region Vectors
    public static readonly LossFunction<double[]> MeanSquaredErrorVector = new(
        (predicted, actual) => {
            if (predicted.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            double sum = 0;
            for (int i = 0; i < predicted.Length; i++)
                sum += Math.Pow(predicted[i] - actual[i], 2);
            return sum / predicted.Length;
        },
        (predicted, actual) => {
            if (predicted.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            double[] gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
                gradient[i] = 2 * (predicted[i] - actual[i]) / predicted.Length;
            return gradient;
        }
    );
    public static readonly LossFunction<double[]> CrossEntropy = new(
        (predicted, actual) => {
            if (predicted.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            double sum = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                if (Math.Abs(actual[i] - 1) < 1e-15)
                    sum -= Math.Log(predicted[i] + 1e-15);
                else
                    sum -= Math.Log(1 - predicted[i] + 1e-15);
            }
            return sum / predicted.Length;
        },
        (predicted, actual) => {
            if (predicted.Length != actual.Length)
                throw new ArgumentException("Vectors must be of the same length.");
            double[] gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
                gradient[i] = (predicted[i] - actual[i]) / ((predicted[i] * (1 - predicted[i])) + 1e-15) / predicted.Length;
            return gradient;
        }
    );
    #endregion
    #region Matrices
    public static readonly LossFunction<double[,]> MeanSquaredErrorMatrix = new(
        (predicted, actual) => {
            if (predicted.GetLength(0) != actual.GetLength(0) || predicted.GetLength(1) != actual.GetLength(1))
                throw new ArgumentException("Matrices must be of the same dimensions.");
            double sum = 0;
            int rows = predicted.GetLength(0), cols = predicted.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    sum += Math.Pow(predicted[i, j] - actual[i, j], 2);
            return sum / (rows * cols);
        },
        (predicted, actual) => {
            if (predicted.GetLength(0) != actual.GetLength(0) || predicted.GetLength(1) != actual.GetLength(1))
                throw new ArgumentException("Matrices must be of the same dimensions.");
            int rows = predicted.GetLength(0), cols = predicted.GetLength(1);
            double[,] gradient = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    gradient[i, j] = 2 * (predicted[i, j] - actual[i, j]) / (rows * cols);
            return gradient;
        }
    );
    #endregion
}

public record struct LossFunction<T>(Func<T, T, double> Function, Func<T, T, T> Derivative);