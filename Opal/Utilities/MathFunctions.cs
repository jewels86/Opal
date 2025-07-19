using Opal.Utilities.ANNs;

namespace Opal.Utilities;

public static class MathFunctions
{
    // TODO: Clean up and organize this class
    public static double[] ReLu(double[] x) => x.Select(v => Math.Max(0, v)).ToArray();
    public static double[] ReLuDerivative(double[] x) => x.Select(v => v > 0 ? 1.0 : 0.0).ToArray();

    public static double[,] InitializeMatrix(int rows, int cols, Func<int, int, double> initializer)
    {
        double[,] matrix = new double[rows,cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = initializer(i, j);
            }
        }

        return matrix;
    }

    public static double[,] RandomMatrix(int rows, int cols)
    {
        return InitializeMatrix(rows, cols, (i, j) => Random.Shared.NextDouble() * 2 - 1);
    }
    public static double[,] ZeroMatrix(int rows, int cols)
    {
        return InitializeMatrix(rows, cols, (i, j) => 0.0);
    }
    
    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    public static double[] Sigmoid(double[] x) => x.Select(Sigmoid).ToArray();
    public static double Tanh(double x) => Math.Tanh(x);
    public static double[] Tanh(double[] x) => x.Select(Tanh).ToArray();
    
    public static double SigmoidDerivative(double x) => Sigmoid(x) * (1 - Sigmoid(x));
    public static double[] SigmoidDerivative(double[] x) => x.Select(SigmoidDerivative).ToArray();
    public static double TanhDerivative(double x) => 1.0 - Math.Pow(Tanh(x), 2);
    public static double[] TanhDerivative(double[] x) => x.Select(TanhDerivative).ToArray();
    
    public static double[] ZeroVector(int length)
    {
        return new double[length];
    }

    public static double[] Apply(ILayer.ActivationFunction fn, double[] x)
    {
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            y[i] = fn(x[i]);
        return y;
    }
    public static double[] Add(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x + y).ToArray();
    }
    public static double[] Subtract(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x - y).ToArray();
    }
    public static double[] Multiply(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return a.Zip(b, (x, y) => x * y).ToArray();
    }
    public static double[] Multiply(double[] a, double scalar)
    {
        return a.Select(x => x * scalar).ToArray();
    }

    public static double[] Multiply(double[,] matrix, double[] vector)
    {
        if (matrix.GetLength(1) != vector.Length) throw new ArgumentException("Matrix columns must match vector length.");
        double[] result = new double[matrix.GetLength(0)];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            result[i] = 0;
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }
        return result;
    }
    public static double[] Multiply(double[] vector, double[,] matrix)
    {
        if (vector.Length != matrix.GetLength(0)) throw new ArgumentException("Vector length must match matrix rows.");
        double[] result = new double[matrix.GetLength(1)];
        for (int j = 0; j < matrix.GetLength(1); j++)
        {
            result[j] = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                result[j] += vector[i] * matrix[i, j];
            }
        }
        return result;
    }
    public static void AddOuterProduct(double[,] mat, double[] vec1, double[] vec2)
    {
        for (int i = 0; i < vec1.Length; i++)
        for (int j = 0; j < vec2.Length; j++)
            mat[i, j] += vec1[i] * vec2[j];
    }

    public static void AddMatVecMul(double[] result, double[,] mat, double[] vec)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        for (int j = 0; j < mat.GetLength(1); j++)
            result[i] += mat[i, j] * vec[j];
    }

    public static void SubtractInPlace(double[,] param, double[,] grad, double lr)
    {
        for (int i = 0; i < param.GetLength(0); i++)
        for (int j = 0; j < param.GetLength(1); j++)
            param[i, j] -= lr * grad[i, j];
    }

    public static void SubtractInPlace(double[] param, double[] grad, double lr)
    {
        for (int i = 0; i < param.Length; i++)
            param[i] -= lr * grad[i];
    }
    public static double[] Divide(double[] a, double scalar)
    {
        if (scalar == 0) throw new DivideByZeroException("Cannot divide by zero.");
        return a.Select(x => x / scalar).ToArray();
    }
    public static double[] Dot(double[] a, double[] b) 
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must be of the same length.");
        return new double[] { a.Zip(b, (x, y) => x * y).Sum() };
    }
    public static void DivideInPlace(double[,] mat, double denom)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        for (int j = 0; j < mat.GetLength(1); j++)
            mat[i, j] /= denom;
    }
    public static void DivideInPlace(double[] vec, double denom)
    {
        for (int i = 0; i < vec.Length; i++)
            vec[i] /= denom;
    }
    public static void ClipInPlace(double[,] mat, double min, double max)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        for (int j = 0; j < mat.GetLength(1); j++)
            mat[i, j] = Clip(mat[i, j], min, max);
    }
    public static void ClipInPlace(double[] vec, double min, double max)
    {
        for (int i = 0; i < vec.Length; i++)
            vec[i] = Clip(vec[i], min, max);
    }
    public static double Clip(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
    public static double[,] GetBatchSample(double[,,] inputSequence, int batchIdx)
    {
        int time = inputSequence.GetLength(1);
        int inputSize = inputSequence.GetLength(2);
        double[,] sample = new double[time, inputSize];
        for (int t = 0; t < time; t++)
        for (int i = 0; i < inputSize; i++)
            sample[t, i] = inputSequence[batchIdx, t, i];
        return sample;
    }
    public static double[] GetInputFromSample(double[,] sample, int timeStep)
    {
        int inputSize = sample.GetLength(1);
        double[] input = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
            input[i] = sample[timeStep, i];
        return input;
    }
}