using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public abstract class FfNetwork<TWeights, TBiases, TInput, THidden, TOutput> : INetwork<TInput, TOutput>
    where TInput : notnull where TOutput : notnull
    where THidden : notnull
    where TWeights : notnull where TBiases : notnull
{
    public LegacyFfLayer<TWeights, TBiases, TInput, THidden> InputLayer { get; }
    public List<LegacyFfLayer<TWeights, TBiases, THidden, THidden>> HiddenLayers { get; }
    public LegacyFfLayer<TWeights, TBiases, THidden, TOutput> OutputLayer { get; }
    
    public string Name { get; private set; }
    
    public int[] InputShape { get; private set; }
    public int[] HiddenShape { get; private set; }
    public int[] OutputShape { get; private set; }
    
    public IFfTensorOperations<TWeights, TBiases, TInput, THidden> InputTensorOperations { get; }
    public IFfTensorOperations<TWeights, TBiases, THidden, THidden> HiddenTensorOperations { get; }
    public IFfTensorOperations<TWeights, TBiases, THidden, TOutput> OutputTensorOperations { get; }
    
    public ActivationFunction<TOutput> OutputActivation { get; }
    public ActivationFunction<THidden> HiddenActivation { get; }
    
    public LossFunction<TOutput> LossFunction { get; }
    public IOptimizer<TWeights, TBiases> Optimizer { get; }
    
    protected FfNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<THidden> hiddenActivation, ActivationFunction<TOutput> outputActivation,
        LossFunction<TOutput> lossFunction, IOptimizer<TWeights, TBiases> optimizer, 
        IFfTensorOperations<TWeights, TBiases, TInput, THidden> inputTensorOperations,
        IFfTensorOperations<TWeights, TBiases, THidden, THidden> hiddenTensorOperations,
        IFfTensorOperations<TWeights, TBiases, THidden, TOutput> outputTensorOperations,
        string name = "FfNetwork")
    {
        InputShape = inputShape;
        HiddenShape = hiddenShape;
        OutputShape = outputShape;
        Name = name;
        
        InputTensorOperations = inputTensorOperations;
        HiddenTensorOperations = hiddenTensorOperations;
        OutputTensorOperations = outputTensorOperations;

        OutputActivation = outputActivation;
        HiddenActivation = hiddenActivation;
        LossFunction = lossFunction;
        Optimizer = optimizer;
        
        InputLayer = new(InputShape, HiddenShape, HiddenActivation, InputTensorOperations, Optimizer);
        HiddenLayers = [];
        for (int i = 0; i < hiddenLayers; i++)
            HiddenLayers.Add(new(HiddenShape, HiddenShape, HiddenActivation, HiddenTensorOperations, Optimizer));
        OutputLayer = new(HiddenShape, OutputShape, OutputActivation, OutputTensorOperations, Optimizer);
    }
    

    public TOutput Forward(TInput input)
    {
        THidden hidden = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            hidden = layer.Forward(hidden);
        return OutputLayer.Forward(hidden);
    }

    public void Train(TInput[] inputs, TOutput[] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var target = targets[i];
                var hidden = InputLayer.Forward(input);
                foreach (var layer in HiddenLayers)
                    hidden = layer.Forward(hidden);
                var output = OutputLayer.Forward(hidden);

                var lossGrad = LossFunction.Derivative(output, target);

                var grad = OutputLayer.Backward(lossGrad, learningRate);
                for (int h = HiddenLayers.Count - 1; h >= 0; h--)
                    grad = HiddenLayers[h].Backward(grad, learningRate);
                InputLayer.Backward(grad, learningRate);
            }
        }
    }

    public double EvaluateLoss(TInput[] inputs, TOutput[] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var predicted = Forward(inputs[i]);
            var actual = targets[i];
            totalLoss += LossFunction.Function(predicted, actual);
        }
        return totalLoss / inputs.Length;
    }
    public void Reset()
    {
        InputLayer.Reset();
        foreach (var layer in HiddenLayers)
            layer.Reset();
        OutputLayer.Reset();
    }

    public virtual void Save(string path)
    {
        BinaryWriter writer = new(File.OpenWrite(path));
        
        BinaryWriting.WriteString(writer, Name);
        
        BinaryWriting.WriteShape(writer, InputShape);
        BinaryWriting.WriteShape(writer, HiddenShape);
        BinaryWriting.WriteShape(writer, OutputShape);
        
        InputLayer.Write(writer);
        writer.Write(HiddenLayers.Count);
        foreach (var layer in HiddenLayers)
            layer.Write(writer);
        OutputLayer.Write(writer);
    }

    public virtual void Load(string path)
    {
        BinaryReader reader = new(File.OpenRead(path));

        Name = BinaryWriting.ReadString(reader);

        InputShape = BinaryWriting.ReadShape(reader);
        HiddenShape = BinaryWriting.ReadShape(reader);
        OutputShape = BinaryWriting.ReadShape(reader);
        
        InputLayer.Read(reader);
        int count = reader.Read();
        for (int i = 0; i < count; i++)
            HiddenLayers.Add(new(InputShape, OutputShape, HiddenActivation, HiddenTensorOperations, Optimizer));
        for (int i = 0; i < count; i++)
            HiddenLayers[i].Read(reader);
        OutputLayer.Read(reader);
    }
}