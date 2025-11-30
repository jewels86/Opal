using Jewels.Lazulite;
using Opal.Utilities;

namespace Opal.NNs;

public class VectorFfNetwork(
    int inputSize,
    int hiddenSize,
    int outputSize,
    int numHiddenLayers,
    Func<Tensor<float[]>, Tensor<float[]>> hiddenActivation,
    Func<Tensor<float[]>, Tensor<float[]>> outputActivation,
    Func<Tensor<float[]>, Value<float[]>, Tensor<float>> lossFunction)
    : FfNetwork<float[], float[], float[,], float[,], float[], float[]>(
        CreateLayer(inputSize, hiddenSize, hiddenActivation),
        CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
        CreateLayer(hiddenSize, outputSize, outputActivation),
        lossFunction,
        hiddenSize,
        hiddenActivation)
{


    protected override FfLayer<float[], float[], float[,], float[]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static FfLayer<float[], float[], float[,], float[]> CreateLayer(
        int inputSize,
        int outputSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation)
    {
        var catalog = new VectorCatalog();
    
        var weights = TensorGeneration.XavierMatrix(outputSize, inputSize).NonDisposable();
        var biases = TensorGeneration.HeVector(outputSize, inputSize).NonDisposable();
    
        return new(weights, biases, activation, catalog);
    }

    private static List<FfLayer<float[], float[], float[,], float[]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation)
    {
        List<FfLayer<float[], float[], float[,], float[]>> layers = [];
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}