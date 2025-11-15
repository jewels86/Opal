using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;

namespace Opal.NNs.Lstm;

public class VectorLstmNetwork : LstmNetwork<double[], double[], double[], double[], double[], double[]>
{
    
    
    public VectorLstmNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        ActivationFunction<double[]> sigmoidActivation,
        ActivationFunction<double[]> tanhActivation,
        LossFunction<double[]> lossFunction,
        string name = "VectorLstmNetwork")
        : base(
            CreateLayer(inputSize, hiddenSize, tanhActivation, sigmoidActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, tanhActivation, sigmoidActivation),
            CreateLayer(hiddenSize, outputSize, tanhActivation, sigmoidActivation),
            lossFunction,
            hiddenSize,
            sigmoidActivation, tanhActivation,
            tanhActivation, sigmoidActivation,
            name)
    {
    }
    
    protected override LstmLayer<double[], double[], double[]> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, TanhHiddenActivation, SigmoidHiddenActivation);

    private static Tensor<double[]>[] CreateWeightArray(int outputSize, int weightSize, Random random)
    {
        var weights = new Tensor<double[]>[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            var weight = new double[weightSize];
            for (int j = 0; j < weightSize; j++)
                weight[j] = random.NextDouble() * 2 - 1;
            weights[i] = new Tensor<double[]>(weight, null, _ => { }, Vectors.Zeros(weightSize));
        }
        return weights;
    }

    private static Tensor<double[]> CreateBiasTensor(int size) => new(Vectors.Zeros(size), null, _ => { }, Vectors.Zeros(size));

    private static LstmLayer<double[], double[], double[]> CreateLayer(
        int inputSize,
        int outputSize,
        ActivationFunction<double[]> tanhActivation,
        ActivationFunction<double[]> sigmoidActivation)
    {
        var catalog = new VectorCatalog();
        var random = new Random();
        
        int encoderConcatSize = inputSize + outputSize;
        var encoderForgetWeights = CreateWeightArray(outputSize, encoderConcatSize, random);
        var encoderInputWeights = CreateWeightArray(outputSize, encoderConcatSize, random);
        var encoderCellWeights = CreateWeightArray(outputSize, encoderConcatSize, random);
        var encoderOutputWeights = CreateWeightArray(outputSize, encoderConcatSize, random);
        
        int decoderConcatSize = outputSize + outputSize;
        var decoderForgetWeights = CreateWeightArray(outputSize, decoderConcatSize, random);
        var decoderInputWeights = CreateWeightArray(outputSize, decoderConcatSize, random);
        var decoderCellWeights = CreateWeightArray(outputSize, decoderConcatSize, random);
        var decoderOutputWeights = CreateWeightArray(outputSize, decoderConcatSize, random);
        
        var encoderForgetBiases = CreateBiasTensor(outputSize);
        var encoderInputBiases = CreateBiasTensor(outputSize);
        var encoderCellBiases = CreateBiasTensor(outputSize);
        var encoderOutputBiases = CreateBiasTensor(outputSize);
        
        var decoderForgetBiases = CreateBiasTensor(outputSize);
        var decoderInputBiases = CreateBiasTensor(outputSize);
        var decoderCellBiases = CreateBiasTensor(outputSize);
        var decoderOutputBiases = CreateBiasTensor(outputSize);
        
        return new LstmLayer<double[], double[], double[]>
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
            SigmoidActivation = sigmoidActivation,
            TanhActivation = tanhActivation,
            Catalog = catalog
        };
    }

    private static List<LstmLayer<double[], double[], double[]>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        ActivationFunction<double[]> tanhActivation,
        ActivationFunction<double[]> sigmoidActivation)
    {
        var layers = new List<LstmLayer<double[], double[], double[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, tanhActivation, sigmoidActivation));
        return layers;
    }
}