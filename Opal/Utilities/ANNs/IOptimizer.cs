using Opal.Mathematics;

namespace Opal.Utilities.ANNs;

public interface IOptimizer<TWeights, TBiases>
{
    TWeights UpdateWeights(TWeights weights, TWeights gradWeights, double learningRate);
    TBiases UpdateBiases(TBiases biases, TBiases gradBiases, double learningRate);
}

public class StandardMatrixOptimizer : IOptimizer<double[,], double[,]>
{
    public double[,] UpdateBiases(double[,] biases, double[,] gradBiases, double learningRate) => Matrices.Subtract(biases, Matrices.Multiply(gradBiases, learningRate));
    public double[,] UpdateWeights(double[,] weights, double[,] gradWeights, double learningRate) => Matrices.Subtract(weights, Matrices.Multiply(gradWeights, learningRate));
}

public class StandardScalarOptimizer : IOptimizer<double, double>
{
    public double UpdateBiases(double biases, double gradBiases, double learningRate) => biases - gradBiases * learningRate;
    public double UpdateWeights(double weights, double gradWeights, double learningRate) => weights - gradWeights * learningRate;
}

public class StandardVectorOptimizer : IOptimizer<double[,], double[]>
{
    public double[] UpdateBiases(double[] biases, double[] gradBiases, double learningRate) => Vectors.Subtract(biases, Vectors.Multiply(gradBiases, learningRate));
    public double[,] UpdateWeights(double[,] weights, double[,] gradWeights, double learningRate) => Matrices.Subtract(weights, Matrices.Multiply(gradWeights, learningRate));
}