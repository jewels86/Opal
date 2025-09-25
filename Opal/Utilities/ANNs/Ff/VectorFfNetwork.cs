using Opal.Mathematics;

namespace Opal.Utilities.ANNs.Ff;

public class VectorFfNetwork : FfNetwork<double[,], double[], double[], double[], double[]>
{
    public VectorFfNetwork(
        int[] inputShape,
        int[] hiddenShape,
        int[] outputShape,
        int hiddenLayers,
        ActivationFunction<double[]>? hiddenActivation = null, 
        ActivationFunction<double[]>? outputActivation = null,
        LossFunction<double[]>? lossFunction = null, 
        IOptimizer<double[,], double[]>? optimizer = null,
        string name = "VectorFfNetwork")
        : base(
            inputShape,
            hiddenShape,
            outputShape,
            hiddenLayers,
            hiddenActivation ?? ActivationFunctions.ReLuVector, 
            outputActivation ?? ActivationFunctions.ReLuVector,
            lossFunction ?? LossFunctions.CrossEntropy, 
            optimizer ?? new StandardVectorOptimizer(),
            new VectorFfTensorOperations(),
            new VectorFfTensorOperations(),
            new VectorFfTensorOperations(),
            name)
    {
    }
}

public class VectorFfTensorOperations : IFfTensorOperations<double[,], double[], double[], double[]>
{
    public double[] Add(double[] output, double[] biases) => Vectors.Add(output, biases);
    public double[] Apply(double[] output, Func<double, double> activation) => Vectors.ApplyElementwise(output, activation);
    public double[] DefaultBiases(int[] shape) => new double[shape[0]];
    public double[] DefaultOutput(int[] shape) => new double[shape[0]];
    public double[,] DefaultWeights(int[] outputShape, int[] inputShape) => Matrices.RandomMatrix(outputShape[0], inputShape[0]);
    public double[] DefaultInput(int[] shape) => new double[shape[0]];
    public double[] Multiply(double[,] weights, double[] input) => Matrices.Multiply(weights, input);
    public double[] GradBiases(double[] gradZ) => gradZ;
    public double[] GradInput(double[,] weights, double[] gradZ)
    {
        return Matrices.Multiply(Matrices.Transpose(weights), gradZ);
    }
    public double[,] GradWeights(double[] gradZ, double[] lastInput) => Vectors.OuterProduct(gradZ, lastInput);
}