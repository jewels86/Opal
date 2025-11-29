using Jewels.Lazulite;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class VectorFfNetwork : FfNetwork<float[], float[], float[,], float[,]>
{
    public VectorFfNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        Func<Tensor<float[]>, Tensor<float[]>> hiddenActivation,
        Func<Tensor<float[,]>, Tensor<float[,]>> hiddenActivationBatched,
        Func<Tensor<float[]>, Tensor<float[]>> outputActivation,
        Func<Tensor<float[,]>, Tensor<float[,]>> outputActivationBatched,
        Func<Tensor<float[]>, Value<float[]>, Tensor<float>> lossFunction,
        Func<Tensor<float[,]>, Value<float[,]>, Tensor<float>> batchedLossFunction)
        : base(
            CreateLayer(inputSize, hiddenSize, hiddenActivation, hiddenActivationBatched),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, hiddenActivation, hiddenActivationBatched),
            CreateLayer(hiddenSize, outputSize, outputActivation, outputActivationBatched),
            lossFunction,
            batchedLossFunction,
            hiddenSize,
            hiddenActivation,
            hiddenActivationBatched)
    {
    }

    public VectorFfNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        Func<ITensor, ITensor> hiddenActivation,
        Func<ITensor, ITensor> outputActivation,
        Func<ITensor, IValue, Tensor<float>> lossFunction)
        : this(
            inputSize,
            hiddenSize,
            outputSize,
            numHiddenLayers,
            input => (Tensor<float[]>)hiddenActivation(input),
            input => (Tensor<float[,]>)hiddenActivation(input),
            input => (Tensor<float[]>)outputActivation(input),
            input => (Tensor<float[,]>)outputActivation(input),
            lossFunction,
            lossFunction)
    {
    }
    

    protected override FfLayer<float[], float[], float[,], float[,]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation, HiddenActivationBatched);

    private static FfLayer<float[], float[], float[,], float[,]> CreateLayer(
        int inputSize,
        int outputSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation,
        Func<Tensor<float[,]>, Tensor<float[,]>> activationBatched)
    {
        var catalog = new VectorCatalog();
    
        var weights = TensorGeneration.XavierMatrix(outputSize, inputSize).NonDisposable();
        var biases = TensorGeneration.HeVector(outputSize, inputSize).NonDisposable();
    
        return new(weights, biases, activation, activationBatched, catalog);
    }

    private static List<FfLayer<float[], float[], float[,], float[,]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<Tensor<float[]>, Tensor<float[]>> activation,
        Func<Tensor<float[,]>, Tensor<float[,]>> activationBatched)
    {
        List<FfLayer<float[], float[], float[,],float[,]>> layers = [];
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation, activationBatched));
        return layers;
    }
}