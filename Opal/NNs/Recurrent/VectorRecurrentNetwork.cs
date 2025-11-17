using Opal.Autograd;
using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Recurrent;

public class VectorRecurrentNetwork : RecurrentNetwork<VectorTensorStorage, VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage, MatrixTensorStorage, MatrixTensorStorage>
{
    public VectorRecurrentNetwork(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int numHiddenLayers,
        Func<VectorTensor, VectorTensor> hiddenActivation,
        Func<VectorTensor, VectorTensor> outputActivation,
        Func<VectorTensor, VectorTensorStorage, ScalarTensor> lossFunction)
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
    
    public double[] Forward(double[] input) => Forward(Operations.NewVector(input, Vectors.Zeros(input.Length))).Value.ToHost();
    
    public double[] ForwardSequence(double[][] sequence) =>
        ForwardSequence(sequence.Select(Operations.NewDefaultVectorStorage).ToArray()).ToHost();
    
    public void TrainSequences(double[][][] sequences, double[][] targets, int epochs, double learningRate) =>
        TrainSequences(
            sequences.Select(seq => seq.Select(Operations.NewDefaultVectorStorage).ToArray()).ToArray(),
            targets.Select(Operations.NewDefaultVectorStorage).ToArray(),
            epochs,
            learningRate);
    
    public double EvaluateLossSequences(double[][][] sequences, double[][] targets) =>
        EvaluateLossSequences(
            sequences.Select(seq => seq.Select(Operations.NewDefaultVectorStorage).ToArray()).ToArray(),
            targets.Select(Operations.NewDefaultVectorStorage).ToArray());

    protected override RecurrentLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateHiddenLayer() =>
        CreateLayer(HiddenSize, HiddenSize, HiddenActivation);

    private static RecurrentLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage> CreateLayer(
        int inputSize,
        int outputSize,
        Func<VectorTensor, VectorTensor> activation)
    {
        var catalog = new VectorCatalog();
    
        MatrixTensor inputWeights = ParameterGeneration.RandomMatrix(1, -1, outputSize, inputSize);
        MatrixTensor recurrentWeights = ParameterGeneration.RandomMatrix(1, -1, outputSize, outputSize);
        VectorTensor biases = ParameterGeneration.GenerateVector(_ => 0.0, outputSize);
        VectorTensor state = ParameterGeneration.GenerateVector(_ => 0.0, outputSize);
    
        return new(inputWeights, recurrentWeights, biases, state, activation, catalog);
    }

    private static List<RecurrentLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>> CreateHiddenLayers(
        int numLayers,
        int hiddenSize,
        Func<VectorTensor, VectorTensor> activation)
    {
        var layers = new List<RecurrentLayer<VectorTensorStorage, VectorTensorStorage, MatrixTensorStorage>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, activation));
        return layers;
    }
}
