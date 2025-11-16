using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;

namespace Opal.NNs.Ff;

public class VectorFfNetwork : FfNetwork<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>, ITensorStorage<double[,]>, ITensorStorage<double[,]>>
{
    public VectorFfNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<ITensorStorage<double[]>> hiddenActivation,
        ActivationFunction<ITensorStorage<double[]>> outputActivation,
        LossFunction<ITensorStorage<double[]>> lossFunction,
        string name = "VectorFfNetwork")
        : base(
            CreateLayer(inputSize, hiddenSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(hiddenSize, outputSize, outputActivation),
            lossFunction,
            hiddenSize,
            hiddenActivation,
            name)
    {
    }

    protected override FfLayer<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static FfLayer<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>> CreateLayer(
        int inputSize,
        int outputSize,
        ActivationFunction<ITensorStorage<double[]>> activation)
    {
        var catalog = new VectorCatalog();
        var _weights = new double[outputSize, inputSize];
    
        var random = new Random();
        for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < inputSize; j++)
            _weights[i, j] = random.NextDouble() * 2 - 1;

        MatrixTensor weights = new(
            Operations.NewDefaultMatrixStorage(_weights), 
            null, _ => { }, 
            Operations.NewDefaultMatrixStorage(Matrices.Zeros(outputSize, inputSize)));
        VectorTensor biases = new(
            Operations.NewDefaultVectorStorage(Vectors.Zeros(outputSize)), 
            null, _ => { }, 
            Operations.NewDefaultVectorStorage(Vectors.Zeros(outputSize)));
    
        return new(weights, biases, activation, catalog);
    }

    private static List<FfLayer<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<ITensorStorage<double[]>> activation)
    {
        var layers = new List<FfLayer<ITensorStorage<double[]>, ITensorStorage<double[]>, ITensorStorage<double[,]>>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}