using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public abstract class LstmNetwork<TIn, THidden, TOut, TWeightIn, TWeightHidden, TWeightOut> : INetwork<TIn, TOut>, ISequentialNetwork<TIn, TOut>
    where TIn : notnull
    where THidden : notnull
    where TOut : notnull
    where TWeightIn : notnull
    where TWeightHidden : notnull
    where TWeightOut : notnull
{
    public LstmLayer<TIn, THidden, TWeightIn> InputLayer { get; set; }
    public List<LstmLayer<THidden, THidden, TWeightHidden>> HiddenLayers { get; set; }
    public LstmLayer<THidden, TOut, TWeightOut> OutputLayer { get; set; }
    
    public string Name { get; set; }
    public LossFunction<TOut> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected ActivationFunction<TOut> SigmoidOutActivation { get; }
    protected ActivationFunction<TOut> TanhOutActivation { get; }
    protected ActivationFunction<THidden> TanhHiddenActivation { get; }
    protected ActivationFunction<THidden> SigmoidHiddenActivation { get; }
    
    
    protected LstmNetwork(
        LstmLayer<TIn, THidden, TWeightIn> inputLayer,
        List<LstmLayer<THidden, THidden, TWeightHidden>> hiddenLayers,
        LstmLayer<THidden, TOut, TWeightOut> outputLayer,
        LossFunction<TOut> lossFunction,
        int hiddenSize,
        ActivationFunction<TOut> sigmoidOutActivation,
        ActivationFunction<TOut> tanhOutActivation,
        ActivationFunction<THidden> tanhHiddenActivation,
        ActivationFunction<THidden> sigmoidHiddenActivation,
        string name = "lstmNetwork")
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
        SigmoidOutActivation = sigmoidOutActivation;
        TanhOutActivation = tanhOutActivation;
        TanhHiddenActivation = tanhHiddenActivation;
        SigmoidHiddenActivation = sigmoidHiddenActivation;
        Name = name;
    }
    
    public TOut Forward(TIn input)
    {
        THidden hidden = InputLayer.Forward(input);
        hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
        return OutputLayer.Forward(hidden);
    }
    
    public TOut ForwardSequence(TIn[] sequence)
    {
        TOut output = default!;
        
        foreach (var input in sequence)
        {
            THidden hidden = InputLayer.Forward(input);
            hidden = HiddenLayers.Aggregate(hidden, (current, layer) => layer.Forward(current));
            output = OutputLayer.Forward(hidden);
        }
        
        return output;
    }
    
    public void Train(TIn[] inputs, TOut[] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inputTensor = new Tensor<TIn>(inputs[i], null, _ => { }, 
                    InputLayer.Catalog.ZeroGradient(inputs[i]));
            
                var hiddenTensor = InputLayer.Forward(inputTensor);
                foreach (var layer in HiddenLayers)
                    hiddenTensor = layer.Forward(hiddenTensor);
                var outputTensor = OutputLayer.Forward(hiddenTensor);

                var lossTensor = LossFunction.Function(outputTensor, targets[i]);

                lossTensor.Backward(1.0);

                InputLayer.UpdateParameters(learningRate);
                foreach (var layer in HiddenLayers)
                    layer.UpdateParameters(learningRate);
                OutputLayer.UpdateParameters(learningRate);
            }
        }
    }
    
    public void TrainSequences(TIn[][] sequences, TOut[] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < sequences.Length; i++)
            {
                Tensor<TOut> outputTensor = null!;
                
                foreach (var input in sequences[i])
                {
                    var inputTensor = new Tensor<TIn>(input, null, _ => { }, 
                        InputLayer.Catalog.ZeroGradient(input));
                
                    var hiddenTensor = InputLayer.Forward(inputTensor);
                    foreach (var layer in HiddenLayers)
                        hiddenTensor = layer.Forward(hiddenTensor);
                    outputTensor = OutputLayer.Forward(hiddenTensor);
                }

                var lossTensor = LossFunction.Function(outputTensor, targets[i]);
                lossTensor.Backward(1.0);

                InputLayer.UpdateParameters(learningRate);
                foreach (var layer in HiddenLayers)
                    layer.UpdateParameters(learningRate);
                OutputLayer.UpdateParameters(learningRate);
            }
        }
    }
    
    public double EvaluateLoss(TIn[] inputs, TOut[] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var inputTensor = new Tensor<TIn>(inputs[i], null, _ => { }, 
                InputLayer.Catalog.ZeroGradient(inputs[i]));
        
            var hiddenTensor = InputLayer.Forward(inputTensor);
            foreach (var layer in HiddenLayers)
                hiddenTensor = layer.Forward(hiddenTensor);
            var outputTensor = OutputLayer.Forward(hiddenTensor);
        
            var lossTensor = LossFunction.Function(outputTensor, targets[i]);
            totalLoss += lossTensor.Value;
        }
        return totalLoss / inputs.Length;
    }
    
    public double EvaluateLossSequences(TIn[][] sequences, TOut[] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < sequences.Length; i++)
        {
            Tensor<TOut> outputTensor = null!;
            
            foreach (var input in sequences[i])
            {
                var inputTensor = new Tensor<TIn>(input, null, _ => { }, 
                    InputLayer.Catalog.ZeroGradient(input));
            
                var hiddenTensor = InputLayer.Forward(inputTensor);
                foreach (var layer in HiddenLayers)
                    hiddenTensor = layer.Forward(hiddenTensor);
                outputTensor = OutputLayer.Forward(hiddenTensor);
            }
        
            var lossTensor = LossFunction.Function(outputTensor, targets[i]);
            totalLoss += lossTensor.Value;
        }
        return totalLoss / sequences.Length;
    }
    
    public void Save(string path)
    {
        using BinaryWriter writer = new(File.OpenWrite(path));
        
        BinaryWriting.WriteString(writer, Name);
        
        InputLayer.Write(writer);
        writer.Write(HiddenLayers.Count);
        foreach (var layer in HiddenLayers)
            layer.Write(writer);
        OutputLayer.Write(writer);
    }

    public void Load(string path)
    {
        using BinaryReader reader = new(File.OpenRead(path));
        
        Name = BinaryWriting.ReadString(reader);
    
        InputLayer.Read(reader);
        int count = reader.ReadInt32();
        HiddenLayers.Clear();
        for (int i = 0; i < count; i++)
        {
            var layer = CreateHiddenLayer();
            layer.Read(reader);
            HiddenLayers.Add(layer);
        }
        OutputLayer.Read(reader);
    }
    
    protected abstract LstmLayer<THidden, THidden, TWeightHidden> CreateHiddenLayer();
}
