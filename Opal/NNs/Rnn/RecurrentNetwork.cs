using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Rnn;

public abstract class RecurrentNetwork<TInput, THidden, TOutput, TWeightIn, TWeightHidden, TWeightOut, TState>
    : INetwork<TInput, TOutput>
    where TInput : notnull where TOutput : notnull where THidden : notnull
    where TWeightIn : notnull where TWeightHidden : notnull where TWeightOut : notnull
    where TState : notnull
{
    public RecurrentLayer<TInput, THidden, TWeightIn, TState> InputLayer { get; }
    public List<RecurrentLayer<THidden, THidden, TWeightHidden, TState>> HiddenLayers { get; }
    public RecurrentLayer<THidden, TOutput, TWeightOut, TState> OutputLayer { get; }
    
    public string Name { get; set; }
    public LossFunction<TOutput> LossFunction { get; }
    
    protected int HiddenSize { get; }
    protected ActivationFunction<THidden> HiddenActivation { get; }
    
    protected RecurrentNetwork(
        RecurrentLayer<TInput, THidden, TWeightIn, TState> inputLayer,
        List<RecurrentLayer<THidden, THidden, TWeightHidden, TState>> hiddenLayers,
        RecurrentLayer<THidden, TOutput, TWeightOut, TState> outputLayer,
        LossFunction<TOutput> lossFunction,
        int hiddenSize,
        ActivationFunction<THidden> hiddenActivation,
        string name = "RnnNetwork")
    {
        InputLayer = inputLayer;
        HiddenLayers = hiddenLayers;
        OutputLayer = outputLayer;
        LossFunction = lossFunction;
        HiddenSize = hiddenSize;
        HiddenActivation = hiddenActivation;
        Name = name;
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
                var inputTensor = new Tensor<TInput>(inputs[i], null, _ => { }, 
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

    public double EvaluateLoss(TInput[] inputs, TOutput[] targets)
    {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var inputTensor = new Tensor<TInput>(inputs[i], null, _ => { }, 
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

    public void ResetState()
    {
        InputLayer.State = new Tensor<TState>(
            InputLayer.Catalog.ZeroGradient(InputLayer.State.Value), 
            null, 
            _ => { }, 
            InputLayer.Catalog.ZeroGradient(InputLayer.State.Value));
        
        foreach (var layer in HiddenLayers)
        {
            layer.State = new Tensor<TState>(
                layer.Catalog.ZeroGradient(layer.State.Value), 
                null, 
                _ => { }, 
                layer.Catalog.ZeroGradient(layer.State.Value));
        }
        
        OutputLayer.State = new Tensor<TState>(
            OutputLayer.Catalog.ZeroGradient(OutputLayer.State.Value), 
            null, 
            _ => { }, 
            OutputLayer.Catalog.ZeroGradient(OutputLayer.State.Value));
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
    
    protected abstract RecurrentLayer<THidden, THidden, TWeightHidden, TState> CreateHiddenLayer();
}
