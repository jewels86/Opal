using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : FfNetwork<ScalarTensorStorage, ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage, VectorTensorStorage, VectorTensorStorage>
{
    public ScalarFfNetwork(
        int inputSize,
        int hiddenSize,
        int numHiddenLayers,
        ActivationFunction<ScalarTensorStorage> hiddenActivation,
        ActivationFunction<ScalarTensorStorage> outputActivation,
        LossFunction<ScalarTensorStorage> lossFunction,
        string name = "ScalarFfNetwork")
        : base(
            CreateLayer(inputSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(hiddenSize, outputActivation),
            lossFunction,
            hiddenSize,
            hiddenActivation,
            name)
    {
    }

    protected override FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenActivation);
    
    private static FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage> CreateLayer(
        int inputSize, 
        ActivationFunction<ScalarTensorStorage> activation)
    {
        var catalog = new ScalarCatalog();

        var weights = ParameterGeneration.RandomVector(1, -1, inputSize);
        var bias = Operations.NewScalar(0.0, 0.0);
        
        return new(weights, bias, activation, catalog);
    }

    private static List<FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<ScalarTensorStorage> activation)
    {
        var layers = new List<FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, activation));
        return layers;
    }
}

