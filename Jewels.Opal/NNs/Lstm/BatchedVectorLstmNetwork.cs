using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public class BatchedVectorLstmNetwork(
    int inputSize,
    int hiddenSize,
    int outputSize,
    int numHiddenLayers,
    Func<Tensor<float[,,]>, Value<float[,,]>, Tensor<float>> lossFunction,
    Initialization weightsInitialization = Initialization.Xavier,
    Initialization biasesInitialization = Initialization.He)
    : LstmNetwork<float[,,], float[,,], float[,], float[,], float[], float[]>(
        CreateLayer(inputSize, hiddenSize, weightsInitialization, biasesInitialization),
        CreateHiddenLayers(numHiddenLayers, hiddenSize, weightsInitialization, biasesInitialization),
        CreateLayer(hiddenSize, outputSize, weightsInitialization, biasesInitialization),
        lossFunction, hiddenSize)
{

    protected override LstmLayer<float[,,], float[,,], float[,], float[]> CreateHiddenLayer() => CreateLayer(HiddenSize, HiddenSize);

    private static LstmLayer<float[,,], float[,,], float[,], float[]> CreateLayer(
        int inputSize,
        int outputSize,
        Initialization weightsInitialization = Initialization.Xavier,
        Initialization biasesInitialization = Initialization.He)
    {
        var catalog = new BatchedVectorCatalog();

        int encoderConcatSize = inputSize + outputSize;
        var encoderForgetWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize);
        var encoderInputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize);
        var encoderCellWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize);
        var encoderOutputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize);

        int decoderConcatSize = outputSize + outputSize;
        var decoderForgetWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize);
        var decoderInputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize);
        var decoderCellWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize);
        var decoderOutputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize);

        var encoderForgetBiases = Operations.GenerateVector(_ => 0, outputSize);
        var encoderInputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);
        var encoderCellBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);
        var encoderOutputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);

        var decoderForgetBiases = Operations.GenerateVector(_ => 0, outputSize);
        var decoderInputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);
        var decoderCellBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);
        var decoderOutputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: inputSize);

        return new LstmLayer<float[,,], float[,,], float[,], float[]>
        {
            EncoderForgetWeights = encoderForgetWeights,
            EncoderInputWeights = encoderInputWeights,
            EncoderCellWeights = encoderCellWeights,
            EncoderOutputWeights = encoderOutputWeights,
            EncoderForgetBiases = encoderForgetBiases,
            EncoderInputBiases = encoderInputBiases,
            EncoderCellBiases = encoderCellBiases,
            EncoderOutputBiases = encoderOutputBiases,
            DecoderForgetWeights = decoderForgetWeights,
            DecoderInputWeights = decoderInputWeights,
            DecoderCellWeights = decoderCellWeights,
            DecoderOutputWeights = decoderOutputWeights,
            DecoderForgetBiases = decoderForgetBiases,
            DecoderInputBiases = decoderInputBiases,
            DecoderCellBiases = decoderCellBiases,
            DecoderOutputBiases = decoderOutputBiases,
            DefaultHidden = Operations.New(new float[outputSize, outputSize, outputSize]),
            DefaultState = Operations.New(new float[outputSize, outputSize, outputSize]),
            Catalog = catalog
        };
    }

    private static List<LstmLayer<float[,,], float[,,], float[,], float[]>> CreateHiddenLayers(
        int numLayers, 
        int hiddenSize,
        Initialization weightsInitialization = Initialization.Xavier,
        Initialization biasesInitialization = Initialization.He)
    {
        var layers = new List<LstmLayer<float[,,], float[,,], float[,], float[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, weightsInitialization, biasesInitialization));
        return layers;
    }
}