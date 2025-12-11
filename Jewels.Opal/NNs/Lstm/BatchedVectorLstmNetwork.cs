using Jewels.Lazulite;

namespace Jewels.Opal.NNs;

public class BatchedVectorLstmNetwork(
    int inputSize,
    int hiddenSize,
    int outputSize,
    int numHiddenLayers,
    Func<Tensor<float[,]>, Value<float[,]>, Tensor<float>> lossFunction,
    Initialization weightsInitialization = Initialization.Xavier,
    Initialization biasesInitialization = Initialization.Zeros,
    bool optimized = true)
    : LstmNetwork<float[,], float[,], float[,], float[,], float[], float[]>(
        CreateLayer(inputSize, hiddenSize, optimized, weightsInitialization, biasesInitialization),
        CreateHiddenLayers(numHiddenLayers, hiddenSize, optimized, weightsInitialization, biasesInitialization),
        CreateLayer(hiddenSize, outputSize, optimized, weightsInitialization, biasesInitialization),
        lossFunction, hiddenSize)
{
    public bool Optimized => optimized;

    protected override LstmLayer<float[,], float[,], float[,], float[]> CreateHiddenLayer() => CreateLayer(HiddenSize, HiddenSize, Optimized);

    private static LstmLayer<float[,], float[,], float[,], float[]> CreateLayer(
        int inputSize,
        int outputSize,
        bool optimized,
        Initialization weightsInitialization = Initialization.Xavier,
        Initialization biasesInitialization = Initialization.Zeros)
    {
        int encoderConcatSize = inputSize + outputSize;
        var encoderForgetWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize).NonDisposable();
        var encoderInputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize).NonDisposable();
        var encoderCellWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize).NonDisposable();
        var encoderOutputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, encoderConcatSize).NonDisposable();

        int decoderConcatSize = outputSize + outputSize;
        var decoderForgetWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize).NonDisposable();
        var decoderInputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize).NonDisposable();
        var decoderCellWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize).NonDisposable();
        var decoderOutputWeights = Operations.GenerateMatrix(weightsInitialization, outputSize, decoderConcatSize).NonDisposable();

        var encoderForgetBiases = Operations.GenerateVector(_ => 1, outputSize).NonDisposable();
        var encoderInputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: encoderConcatSize).NonDisposable();
        var encoderCellBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: encoderConcatSize).NonDisposable();
        var encoderOutputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: encoderConcatSize).NonDisposable();

        var decoderForgetBiases = Operations.GenerateVector(_ => 1, outputSize).NonDisposable();
        var decoderInputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: decoderConcatSize).NonDisposable();
        var decoderCellBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: decoderConcatSize).NonDisposable();
        var decoderOutputBiases = Operations.GenerateVector(biasesInitialization, outputSize, fanIn: decoderConcatSize).NonDisposable();

        var catalog = new BatchedVectorCatalog();
        
        if (!optimized) return new LstmLayer<float[,], float[,], float[,], float[]>
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
            DefaultHidden = Operations.New(new float[outputSize, outputSize]).NonDisposable(),
            DefaultState = Operations.New(new float[outputSize, outputSize]).NonDisposable(),
            Catalog = new BatchedVectorCatalog()
        };

        return new OptimizedLstmLayer<float[,], float[,], float[,], float[]>
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
            DefaultHidden = Operations.New(new float[outputSize, outputSize]).NonDisposable(),
            DefaultState = Operations.New(new float[outputSize, outputSize]).NonDisposable(),
            Catalog = catalog,
            OptimizedCatalog = catalog
        };
    }

    private static List<LstmLayer<float[,], float[,], float[,], float[]>> CreateHiddenLayers(
        int numLayers, 
        int hiddenSize,
        bool optimized,
        Initialization weightsInitialization = Initialization.Xavier,
        Initialization biasesInitialization = Initialization.He)
    {
        var layers = new List<LstmLayer<float[,], float[,], float[,], float[]>>();
        for (int i = 0; i < numLayers; i++)
            layers.Add(CreateLayer(hiddenSize, hiddenSize, optimized, weightsInitialization, biasesInitialization));
        return layers;
    }

    public Tensor<float[,,]> ForwardSequence(Tensor<float[,,]> sequences)
    {
        var (batch, seqLength, features) = (sequences.Value.Shape[0], 
            sequences.Value.Shape[1], sequences.Value.Shape[2]);
    
        var result = Operations.New(new float[batch, seqLength, outputSize]);
        
        var hidden = Operations.New(Operations.Fill(0f, outputSize, outputSize));
        var state = Operations.New(Operations.Fill(0f, outputSize, outputSize));
    
        for (int t = 0; t < seqLength; t++)
        {
            var timestepInput = Operations.GetSlice(sequences, t);
            var (output, newState) = ForwardWithState(timestepInput, hidden, state);
        
            result = Operations.SetSlice(result, output, t);

            hidden = output;
            state = newState;
        }
    
        return result;
    }

    public Tensor<float[,]> ForwardSequenceFinal(Tensor<float[,,]> sequences)
    {
        var (batch, seqLength, features) = (sequences.Value.Shape[0], 
            sequences.Value.Shape[1], sequences.Value.Shape[2]);
    
        var hidden = Operations.New(Operations.Fill(0f, batch, HiddenSize));
        var state = Operations.New(Operations.Fill(0f, batch, HiddenSize));

        for (int t = 0; t < seqLength; t++)
        {
            var timestepInput = Operations.GetSlice(sequences, t);
            Console.WriteLine($"Hidden shape: {Operations.ToString(hidden.Value.Shape)}");
            Console.WriteLine($"State shape: {Operations.ToString(state.Value.Shape)}");
            Console.WriteLine($"Timestep input shape: {Operations.ToString(timestepInput.Value.Shape)}");
            var (output, newState) = ForwardWithState(timestepInput, hidden, state);

            hidden = output;
            state = newState;
        }
        
        return hidden;
    }
    
    public void Train(Tensor<float[,,]> sequences, Tensor<float[,,]> targets, Func<Tensor<float[,,]>, Value<float[,,]>, Tensor<float>> loss, int epochs, float lr) => 
        Operations.Train<float[,,], float[,,]>(ForwardSequence, loss,  () => UpdateParameters(lr), [sequences], [targets], epochs);
    
    public float EvaluateLoss(Tensor<float[,,]> sequences, Tensor<float[,,]> targets, Func<Tensor<float[,,]>, Value<float[,,]>, Tensor<float>> loss) =>
        Operations.EvaluateLoss<float[,,], float[,,]>(ForwardSequence, loss, [sequences], [targets]);

    public void TrainFinal(Tensor<float[,,]> sequences, Tensor<float[,]> targets, Func<Tensor<float[,]>, Value<float[,]>, Tensor<float>> loss, int epochs, float lr)
    {
        List<ITensor> tensors = [];
        tensors.AddRange(InputLayer.Parameters);
        foreach (var hidden in HiddenLayers) tensors.AddRange(hidden.Parameters);
        tensors.AddRange(OutputLayer.Parameters);
        Operations.Train<float[,,], float[,]>(ForwardSequenceFinal, loss, () => UpdateParameters(lr, DefaultGradClipNorm, tensors), [sequences.Value], [targets.Value], epochs);
    }

    public float EvaluateLossFinal(Tensor<float[,,]> sequences, Tensor<float[,]> targets, Func<Tensor<float[,]>, Value<float[,]>, Tensor<float>> loss)
        => Operations.EvaluateLoss<float[,,], float[,]>(ForwardSequenceFinal, loss, [sequences], [targets]);
}