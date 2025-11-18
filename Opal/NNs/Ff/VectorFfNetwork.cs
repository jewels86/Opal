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
        Func<VectorTensor, VectorTensor> hiddenActivation,
        Func<VectorTensor, VectorTensor> outputActivation,
        Func<VectorTensor, VectorTensorStorage, ScalarTensor> lossFunction,
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
    
    public double[] Forward(double[] input) => Forward(Operations.NewVector(input, Vectors.Zeros(input.Length))).Value.ToHost();

    protected override FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateLayer(
        int inputSize,
        int outputSize,
        Func<VectorTensor, VectorTensor> activation)
    {
        var catalog = new VectorCatalog();
    
        MatrixTensor weights = ParameterGeneration.XavierMatrix(outputSize, inputSize);
        VectorTensor biases = ParameterGeneration.HeVector(outputSize, inputSize);
    
        return new(weights, biases, activation, catalog);
    }

    private static List<FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<VectorTensor, VectorTensor> activation)
    {
        var layers = new List<FfLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}