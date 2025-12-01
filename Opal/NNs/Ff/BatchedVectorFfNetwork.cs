using Jewels.Lazulite;

namespace Opal.NNs;

public class BatchedVectorFfNetwork : FfNetwork<float[,], float[,], float[,], float[,], float[], float[]>
{
    public BatchedVectorFfNetwork(int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        int batchSize,
        Func<Tensor<float[,]>, Tensor<float[,]>> hiddenActivation,
        Func<Tensor<float[,]>, Tensor<float[,]>> outputActivation,
        Func<Tensor<float[,]>, Value<float[,]>, Tensor<float>> lossFunction,
        Initialization weightsInitialization = Initialization.Random,
        Initialization biasesInitialization = Initialization.Random) : base(
        CreateLayer(inputSize, hiddenSize, hiddenActivation, weightsInitialization, biasesInitialization),
        CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation, weightsInitialization, biasesInitialization),
        CreateLayer(hiddenSize, outputSize, outputActivation, weightsInitialization, biasesInitialization),
        lossFunction,
        hiddenSize,
        hiddenActivation)
    {
        DefaultInitialGradient = 1 / (float)batchSize;
    }


    protected override FfLayer<float[,], float[,], float[,], float[]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);
    
    private static FfLayer<float[,], float[,], float[,], float[]> CreateLayer(
        int inputSize,
        int outputSize,
        Func<Tensor<float[,]>, Tensor<float[,]>> activation,
        Initialization weightsInitialization = Initialization.Zeros,
        Initialization biasesInitialization = Initialization.Zeros)
    {
        var catalog = new BatchedVectorCatalog();
    
        var weights = Operations.GenerateMatrix(weightsInitialization, outputSize, inputSize).NonDisposable();
        var biases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize).NonDisposable();
    
        return new(weights, biases, activation, catalog);
    }
    
    private static List<FfLayer<float[,], float[,], float[,], float[]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<Tensor<float[,]>, Tensor<float[,]>> activation,
        Initialization weightsInitialization = Initialization.Zeros,
        Initialization biasesInitialization = Initialization.Zeros)
    {
        List<FfLayer<float[,], float[,], float[,], float[]>> layers = [];
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation, weightsInitialization, biasesInitialization));
        return layers;
    }
}