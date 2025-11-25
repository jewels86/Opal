using Jewels.Lazulite;

using Opal.Utilities;

namespace Opal.NNs.Recurrent;

public class VectorRecurrentNetwork : RecurrentNetwork<float[], float[], float[], float[,], float[,], float[,]>
{
    public VectorRecurrentNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        Func<Tensor<float[]>, Tensor<float[]>> hiddenActivation,
        Func<Tensor<float[]>, Tensor<float[]>> outputActivation,
        Func<Tensor<float[]>, Value<float[]>, Tensor<float>> lossFunction)
        : base(
            hiddenSize,
            CreateLayer(inputSize, hiddenSize, hiddenActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation),
            CreateLayer(hiddenSize, outputSize, outputActivation),
            lossFunction,
            outputActivation,
            hiddenActivation)
    {
    }

    protected override RecurrentLayer<float[], float[], float[,]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static RecurrentLayer<float[], float[], float[,]> CreateLayer(
        int inputSize,
        int outputSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation)
    {
        var catalog = new VectorCatalog();
    
        Tensor<float[,]> inputWeights = TensorGeneration.RandomMatrix(1, -1, outputSize, inputSize);
        Tensor<float[,]> recurrentWeights = TensorGeneration.RandomMatrix(1, -1, outputSize, outputSize);
        Tensor<float[]> biases = TensorGeneration.GenerateVector(_ => 0, outputSize);
        Tensor<float[]> state = TensorGeneration.GenerateVector(_ => 0, outputSize);
    
        return new(inputWeights, recurrentWeights, biases, state, activation, catalog);
    }

    private static List<RecurrentLayer<float[], float[], float[,]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation)
    {
        List<RecurrentLayer<float[], float[], float[,]>> layers = [];
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}
