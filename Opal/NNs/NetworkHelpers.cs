
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Opal.NNs;

public static class NetworkHelpers
{
    public static void Train<TIn, TOut>(
        Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss, Action update,
        Value<TIn>[] inputs, Value<TOut>[] targets, int epochs)
        where TIn : notnull where TOut : notnull
    {
        int aidx = inputs[0].AcceleratorIndex;
        var one = Compute.Get(aidx, 1);
        one.CopyFromCPU([1.0f]);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                using var inputTensor = new Tensor<TIn>(inputs[i], inputs[i].Zeros());
                using var outputTensor = forward(inputTensor);
                using var lossTensor = loss(outputTensor, targets[i]);
                lossTensor.Backward(one);
                update();
            }
            Compute.Flush(aidx);
        }
    }

    public static float EvaluateLoss<TIn, TOut>(
        Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss,
        Value<TIn>[] inputs, Value<TOut>[] targets)
        where TIn : notnull where TOut : notnull
    {
        int aidx = inputs[0].AcceleratorIndex;
        using var totalLoss = new ScalarValue(0, aidx);
        for (int i = 0; i < inputs.Length; i++)
        {
            using var inputTensor = new Tensor<TIn>(inputs[i], inputs[i].Zeros());
            using var outputTensor = forward(inputTensor);
            using var lossTensor = loss(outputTensor, targets[i]);
            totalLoss.UpdateWith(totalLoss + lossTensor.Value.AsScalar());
            Compute.Flush(aidx);
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