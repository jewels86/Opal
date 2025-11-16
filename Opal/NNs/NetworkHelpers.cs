using Opal.Autograd;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs;

public static class NetworkHelpers
{
    public static void Train<TIn, TOut>(
        Func<TIn, TIn> zeroInput, Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, TOut, ScalarTensor> lossFunction, Action updateParameters,
        TIn[] inputs, TOut[] targets, int epochs)
    where TIn : notnull where TOut : notnull
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inputTensor = new Tensor<TIn>(inputs[i], null, _ => { }, 
                    zeroInput(inputs[i]));
            
                var outputTensor = forward(inputTensor);
                var lossTensor = lossFunction(outputTensor, targets[i]);
                lossTensor.Backward(Operations.NewDefaultScalarStorage(1.0));
            
                updateParameters();
            }
        }
    }
    
    public static double EvaluateLoss<TIn, TOut>(
        Func<TIn, TIn> zeroInput, Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, TOut, ScalarTensor> lossFunction,
        TIn[] inputs, TOut[] targets)
    where TIn : notnull where TOut : notnull
    {
        ScalarTensorStorage totalLoss = Operations.NewDefaultScalarStorage(0.0);
        for (int i = 0; i < inputs.Length; i++)
        {
            var inputTensor = new Tensor<TIn>(inputs[i], null, _ => { }, 
                zeroInput(inputs[i]));
        
            var outputTensor = forward(inputTensor);
        
            var lossTensor = lossFunction(outputTensor, targets[i]);
            totalLoss = Operations.AddStorage(totalLoss, lossTensor.Value);
        }
        return totalLoss.ToHost() / inputs.Length;
    }
    
    public static void Save<TIn, THidden, TOut>(
        ILayer<TIn, THidden> inputLayer, List<ILayer<THidden, THidden>> hiddenLayers, ILayer<THidden, TOut> outputLayer, 
        string path)
    where TIn : notnull where THidden : notnull where TOut : notnull
    {
        using BinaryWriter writer = new(File.OpenWrite(path));
        
        inputLayer.Write(writer);
        writer.Write(hiddenLayers.Count);
        foreach (var layer in hiddenLayers)
            layer.Write(writer);
        outputLayer.Write(writer);
    }

    public static void Load<TIn, THidden, TOut>(
        ILayer<TIn, THidden> inputLayer, List<ILayer<THidden, THidden>> hiddenLayers, ILayer<THidden, TOut> outputLayer, Func<ILayer<THidden, THidden>> createHiddenLayer,
        string path)
        where TIn : notnull where THidden : notnull where TOut : notnull
    {
        using BinaryReader reader = new(File.OpenRead(path));
        
        inputLayer.Read(reader);
        int count = reader.ReadInt32();
        hiddenLayers.Clear();
        for (int i = 0; i < count; i++)
        {
            var layer = createHiddenLayer();
            layer.Read(reader);
            hiddenLayers.Add(layer);
        }
        outputLayer.Read(reader);
    }
}