using System.Numerics;
using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public class VectorLstmNetwork : LstmNetwork<VectorTensorStorage, VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage, MatrixTensorStorage, MatrixTensorStorage>
{
    public VectorLstmNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        Func<VectorTensor, VectorTensor> sigmoidActivation,
        Func<VectorTensor, VectorTensor> tanhActivation,
        Func<VectorTensor, VectorTensorStorage, ScalarTensor> lossFunction)
        : base(
            CreateLayer(inputSize, hiddenSize, tanhActivation, sigmoidActivation),
            CreateHiddenLayers(numHiddenLayers, hiddenSize, tanhActivation, sigmoidActivation),
            CreateLayer(hiddenSize, outputSize, tanhActivation, sigmoidActivation),
            lossFunction,
            hiddenSize,
            sigmoidActivation, tanhActivation,
            tanhActivation, sigmoidActivation)
    {
    }
    
    protected override LstmLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, TanhHiddenActivation, SigmoidHiddenActivation);

    private static MatrixTensor CreateWeightArray(int outputSize, int weightSize) => ParameterGeneration.XavierMatrix(outputSize, weightSize);

    private static VectorTensor CreateBiasTensor(int size) =>  Operations.Fill(size, 0.0, 0.0);


    private static LstmLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateLayer(
        int inputSize,
        int outputSize,
        Func<VectorTensor, VectorTensor> tanhActivation,
        Func<VectorTensor, VectorTensor> sigmoidActivation)
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
        
        var encoderForgetBiases = Operations.Fill(outputSize, 1.0, 0.0);
        var encoderInputBiases = CreateBiasTensor(outputSize);
        var encoderCellBiases = CreateBiasTensor(outputSize);
        var encoderOutputBiases = CreateBiasTensor(outputSize);
        
        var decoderForgetBiases = Operations.Fill(outputSize, 1.0, 0.0);
        var decoderInputBiases = CreateBiasTensor(outputSize);
        var decoderCellBiases = CreateBiasTensor(outputSize);
        var decoderOutputBiases = CreateBiasTensor(outputSize);
        
        return new LstmLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>
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
            DefaultHidden = Operations.NewVector(Vectors.Zeros(outputSize)),
            DefaultState = Operations.NewVector(Vectors.Zeros(outputSize)),
            Catalog = catalog
        };
    }

    private static List<LstmLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<VectorTensor, VectorTensor> tanhActivation,
        Func<VectorTensor, VectorTensor> sigmoidActivation)
    {
        var layers = new List<LstmLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, tanhActivation, sigmoidActivation));
        return layers;
    }
}