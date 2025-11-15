﻿using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Lstm;

public class LstmNetwork<TWeights, TBiases, TTensor> : INetwork<TTensor[], TTensor[]>
    where TWeights : notnull
    where TBiases : notnull
    where TTensor : notnull
{
    public string Name { get; }
    
    public int[] InputShape { get; }
    public int[] HiddenShape { get; }
    public int[] OutputShape { get; }
    
    public LegacyLstmLayer<TWeights, TBiases, TTensor> InputLayer { get; }
    public List<LegacyLstmLayer<TWeights, TBiases, TTensor>> HiddenLayers { get; }
    public LegacyLstmLayer<TWeights, TBiases, TTensor> OutputLayer { get; }
    
    protected readonly ILstmTensorOperations<TWeights, TBiases, TTensor> TensorOperations;
    protected readonly IOptimizer<TWeights, TBiases> Optimizer;
    protected readonly ActivationFunction<TTensor> SigmoidActivation;
    protected readonly ActivationFunction<TTensor> TanhActivation;
    protected readonly LossFunction<TTensor[]> LossFunction;

    public LstmNetwork(int[] inputShape, int[] hiddenShape, int[] outputShape, int hiddenLayers,
        ActivationFunction<TTensor> sigmoidActivation, ActivationFunction<TTensor> tanhActivation,
        LossFunction<TTensor[]> lossFunction, IOptimizer<TWeights, TBiases> optimizer,
        ILstmTensorOperations<TWeights, TBiases, TTensor> tensorOperations,
        string name = "lstm network")
    {
        Name = name;
        
        InputShape = inputShape;
        HiddenShape = hiddenShape;
        OutputShape = outputShape;
        
        this.TensorOperations = tensorOperations;
        this.Optimizer = optimizer;
        this.SigmoidActivation = sigmoidActivation;
        this.TanhActivation = tanhActivation;
        this.LossFunction = lossFunction;

        InputLayer = new(inputShape, hiddenShape, hiddenShape, tensorOperations, optimizer, sigmoidActivation,
            tanhActivation);
        HiddenLayers = [];
        for (int i = 0; i < hiddenLayers; i++)
            HiddenLayers.Add(new(hiddenShape, hiddenShape, hiddenShape, tensorOperations, optimizer, sigmoidActivation,
                tanhActivation));
        OutputLayer = new(hiddenShape, hiddenShape, outputShape, tensorOperations, optimizer, sigmoidActivation, tanhActivation);
    }
    
    public virtual TTensor[] Forward(TTensor[] input)
    {
        var output = InputLayer.Forward(input);
        foreach (var layer in HiddenLayers)
            output = layer.Forward(output);
        return OutputLayer.Forward(output);
    }
    
    public virtual void Train(TTensor[][] inputs, TTensor[][] targets, int epochs, double learningRate)
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

    public double EvaluateLoss(TTensor[][] inputs, TTensor[][] targets)
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

    public void Save(string path)
    {
        using var writer = new BinaryWriter(File.Open(path, FileMode.Create));
        
        BinaryWriting.WriteString(writer, Name);
        BinaryWriting.WriteShape(writer, InputShape);
        BinaryWriting.WriteShape(writer, HiddenShape);
        BinaryWriting.WriteShape(writer, OutputShape);
        writer.Write(HiddenLayers.Count);
        
        InputLayer.Write(writer);
        foreach (var layer in HiddenLayers)
            layer.Write(writer);
        OutputLayer.Write(writer);
    }

    public void Load(string path)
    {
        using var reader = new BinaryReader(File.Open(path, FileMode.Open));
        
        string name = BinaryWriting.ReadString(reader);
        int[] inputShape = BinaryWriting.ReadShape(reader);
        int[] hiddenShape = BinaryWriting.ReadShape(reader);
        int[] outputShape = BinaryWriting.ReadShape(reader);
        int hiddenLayerCount = reader.ReadInt32();
        
        if (!inputShape.SequenceEqual(InputShape) || 
            !hiddenShape.SequenceEqual(HiddenShape) || 
            !outputShape.SequenceEqual(OutputShape) ||
            hiddenLayerCount != HiddenLayers.Count)
        {
            throw new InvalidOperationException("Network architecture mismatch. Cannot load weights.");
        }
        
        InputLayer.Read(reader);
        foreach (var layer in HiddenLayers)
            layer.Read(reader);
        OutputLayer.Read(reader);
    }
}
