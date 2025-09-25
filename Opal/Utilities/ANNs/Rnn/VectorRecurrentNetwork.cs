using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Rnn;

public class VectorRecurrentNetwork : RecurrentNetwork<double[,], double[], double[], double[], double[], double[]>
{
    public VectorRecurrentNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null,
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null,
        IOptimizer<double[,], double[]>? optimizer = null,
        string name = "VectorRecurrentNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector,
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.CrossEntropy,
            optimizer ?? new StandardVectorOptimizer(),
            new VectorRecurrentTensorOperations(),
            new VectorRecurrentTensorOperations(),
            new VectorRecurrentTensorOperations(),
            name)
    {
    }
}

public class VectorRecurrentTensorOperations : IRecurrentTensorOperations<double[,], double[], double[], double[], double[]>
{
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[] DefaultBiases(int[] outputShape) => new double[outputShape[0]];
    public double[] DefaultState(int[] outputsShape) => new double[outputsShape[0]];

    public double[] Add(double[] a, double[] b) => Vectors.Add(a, b);
    public double[,] Add(double[,] a, double[,] b)
    {
        int rows = a.GetLength(0);
        int cols = a.GetLength(1);
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = a[i, j] + b[i, j];
        return result;
    }

    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] Multiply(double[] a, double[] b) => Vectors.Multiply(a, b);

    public double[,] GradInputWeights(double[] gradZ, double[] input) => Vectors.OuterProduct(gradZ, input);
    public double[,] GradRecurrentWeights(double[] gradZ, double[] state) => Vectors.OuterProduct(gradZ, state);
    public double[] GradBiases(double[] gradZ) => gradZ;
    public double[] GradOutput(double[,] weights, double[] gradZ) => 
        Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    
    public double[] GradInput(double[,] weights, double[] gradZ) => 
        Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    
    public double[] UpdateState(double[] output) => output;
}