using Jewels.Lazulite;
namespace Jewels.Opal.NNs;

public class VectorLstmNetwork(int inputSize,
    int hiddenSize,
    int outputSize,
    int numHiddenLayers,
    Func<Tensor<float[]>, Value<float[]>, Tensor<float>> lossFunction)
    : LstmNetwork<float[], float[], float[,], float[,], float[], float[]>(
        CreateLayer(inputSize, hiddenSize),
        CreateHiddenLayers(numHiddenLayers, hiddenSize),
        CreateLayer(hiddenSize, outputSize),
        lossFunction, hiddenSize)
{

    protected override LstmLayer<float[], float[], float[,], float[]> CreateHiddenLayer() => CreateLayer(HiddenSize, HiddenSize);

    private static Tensor<float[,]> CreateWeightArray(int outputSize, int weightSize) => Operations.XavierMatrix(outputSize, weightSize);

    private static Tensor<float[]> CreateBiasTensor(int size) =>  Operations.HeVector(size, size);


    private static LstmLayer<float[], float[], float[,], float[]> CreateLayer(
        int inputSize,
        int outputSize)
    {
        var catalog = new VectorCatalog();
        
        int encoderConcatSize = inputSize + outputSize;
        var encoderForgetWeights = CreateWeightArray(outputSize, encoderConcatSize);
        var encoderInputWeights = CreateWeightArray(outputSize, encoderConcatSize);
        var encoderCellWeights = CreateWeightArray(outputSize, encoderConcatSize);
        var encoderOutputWeights = CreateWeightArray(outputSize, encoderConcatSize);
        
        int decoderConcatSize = outputSize + outputSize;
        var decoderForgetWeights = CreateWeightArray(outputSize, decoderConcatSize);
        var decoderInputWeights = CreateWeightArray(outputSize, decoderConcatSize);
        var decoderCellWeights = CreateWeightArray(outputSize, decoderConcatSize);
        var decoderOutputWeights = CreateWeightArray(outputSize, decoderConcatSize);
        
        var encoderForgetBiases = Operations.GenerateVector(_ => 0, outputSize);
        var encoderInputBiases = CreateBiasTensor(outputSize);
        var encoderCellBiases = CreateBiasTensor(outputSize);
        var encoderOutputBiases = CreateBiasTensor(outputSize);
        
        var decoderForgetBiases = Operations.GenerateVector(_ => 0, outputSize);
        var decoderInputBiases = CreateBiasTensor(outputSize);
        var decoderCellBiases = CreateBiasTensor(outputSize);
        var decoderOutputBiases = CreateBiasTensor(outputSize);
        
        return new LstmLayer<float[], float[], float[,], float[]>
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
            DefaultHidden = Operations.New(Operations.Fill(0, outputSize)),
            DefaultState = Operations.New(Operations.Fill(0, outputSize)),
            Catalog = catalog
        };
    }

    private static List<LstmLayer<float[], float[], float[,], float[]>> CreateHiddenLayers(int numLayers, int hiddenSize)
    {
        var layers = new List<LstmLayer<float[], float[], float[,], float[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize));
        return layers;
    }
}