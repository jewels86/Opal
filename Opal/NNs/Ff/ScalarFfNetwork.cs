using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : FfNetwork<ScalarTensorStorage, ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage, VectorTensorStorage, VectorTensorStorage>
{
    public ScalarFfNetwork(
        int numHiddenLayers,
        Func<ScalarTensor, ScalarTensor> hiddenActivation,
        Func<ScalarTensor, ScalarTensor> outputActivation,
        Func<ScalarTensor, ScalarTensorStorage, ScalarTensor> lossFunction,
        string name = "ScalarFfNetwork")
        : base(
            CreateLayer(1, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, 1, hiddenActivation),
            CreateLayer(1, outputActivation),
            lossFunction,
            1,
            hiddenActivation,
            name)
    {
    }

    protected override FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenActivation);
    
    public double Forward(double input) => Forward(Operations.NewScalar(input, 0.0)).Value.ToHost();
    
    private static FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage> CreateLayer(
        int inputSize, 
        Func<ScalarTensor, ScalarTensor> activation)
    {
        var catalog = new ScalarCatalog();

        var weights = ParameterGeneration.RandomVector(1, -1, inputSize);
        var bias = Operations.NewScalar(0.0, 0.0);
        
        return new(weights, bias, activation, catalog);
    }

    private static List<FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<ScalarTensor, ScalarTensor> activation)
    {
        var layers = new List<FfLayer<ScalarTensorStorage, ScalarTensorStorage, VectorTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, activation));
        return layers;
    }
}

