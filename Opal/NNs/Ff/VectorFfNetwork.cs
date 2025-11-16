using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class VectorFfNetwork : FfNetwork<VectorTensorStorage, VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage, MatrixTensorStorage, MatrixTensorStorage>
{
    public VectorFfNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<VectorTensorStorage> hiddenActivation,
        ActivationFunction<VectorTensorStorage> outputActivation,
        LossFunction<VectorTensorStorage> lossFunction,
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

    protected override FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateLayer(
        int inputSize,
        int outputSize,
        ActivationFunction<VectorTensorStorage> activation)
    {
        var catalog = new VectorCatalog();
    
        MatrixTensor weights = ParameterGeneration.RandomMatrix(1, -1, outputSize, inputSize);
        VectorTensor biases = ParameterGeneration.GenerateVector(_ => 0.0, outputSize);
    
        return new(weights, biases, activation, catalog);
    }

    private static List<FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<VectorTensorStorage> activation)
    {
        var layers = new List<FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}