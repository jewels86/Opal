using Jewels.Lazulite;

namespace Opal.NNs;

public static class NetworkHelpers
{
    private static Compute compute => Compute.Instance;
    
    #region Forward
    public static Tensor<TOut> ForwardSequence<TIn, TOut>(
        Action resetState, Func<Tensor<TIn>, Tensor<TOut>> forward,
        Tensor<TIn>[] sequence)
        where TIn : notnull where TOut : notnull
    {
        resetState();
        Tensor<TOut> output = null!;
        foreach (var input in sequence)
        {
            output.Dispose();
            output = forward(input);
        }
        return output;
    }
    #endregion
    #region Training
    public static void Train<TIn, TOut>(
        Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss, Action update,
        Value<TIn>[] inputs, Value<TOut>[] targets, int epochs)
        where TIn : notnull where TOut : notnull
    {
        foreach (var input in inputs) input.NonDisposable();
        foreach (var target in targets) target.NonDisposable();
        
        int aidx = inputs[0].AcceleratorIndex;
        var one = new ScalarValue(compute.Make(aidx, 1, 1)).NonDisposable();
        
        for (int epoch = 0; epoch < epochs; epoch++)
        { 
            // at this point on epoch 1, compute._pool[2][0] has 69 total buffers
            // however when .Distinct is called it drops to 65
            // Compute.Instance._pool[2][1].Count()
            // Compute.Instance._pool[2][1].Distinct().Count()
            
            // doing this with laz 1.3.10 its now 57 and 57
            for (int i = 0; i < inputs.Length; i++)
            {
                using var inputTensor = new Tensor<TIn>(inputs[i], inputs[i].Zeros()); 
                using var outputTensor = forward(inputTensor); // this takes 18 buffers (28 -> 10)
                // doing this with laz 1.3.10 i got a System.InvalidOperationException: Unknown parent accelerator
                // at ILGPU.Runtime.ArrayViewExtensions.GetAccelerator[TView](TView view)
                // when we call accelerator index on one of them- i dont get it
                using var lossTensor = loss(outputTensor, targets[i]); // this takes 2 buffers (10 -> 8)
                lossTensor.Backward(one); // this takes 6 buffers (8 -> 2)
                update(); // this zeroed it out- (2 -> 0)
            }
            compute.Flush(aidx);
        }
    }

    public static void TrainSequences<TIn, TOut>(Func<Tensor<TIn>[], Tensor<TOut>> forward, 
        Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss, Action reset, Action update,
        Value<TIn>[][] inputs, Value<TOut>[] targets, int epochs)
        where TIn : notnull where TOut : notnull
    {
        int aidx = inputs[0][0].AcceleratorIndex;
        var one = new ScalarValue(compute.Make(aidx, 1, 1));
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                reset();
                using var outputTensor = forward(inputs[i].Select(t => new Tensor<TIn>(t, t.Zeros())).ToArray());
                using var lossTensor = loss(outputTensor, targets[i]);
                lossTensor.Backward(one);
                update();
            }
            compute.Flush(aidx);
        }
    }
    #endregion

    #region Evaluation
    public static float EvaluateLoss<TIn, TOut>(
        Func<Tensor<TIn>, Tensor<TOut>> forward, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss,
        Value<TIn>[] inputs, Value<TOut>[] targets)
        where TIn : notnull where TOut : notnull
    {
        int aidx = inputs[0].AcceleratorIndex;
        using var totalLoss = new ScalarValue(0, aidx);
        for (int i = 0; i < inputs.Length; i++)
        {
            using var inputTensor = new Tensor<TIn>(inputs[i].NonDisposable(), inputs[i].Zeros());
            using var outputTensor = forward(inputTensor);
            using var lossTensor = loss(outputTensor, targets[i]);
            totalLoss.UpdateWith(totalLoss + lossTensor.Value.AsScalar());
            compute.Flush(aidx);
        }
        return totalLoss.ToHost() / inputs.Length;
    }
    
    public static float EvaluateLossSequences<TIn, TOut>(
        Func<Tensor<TIn>[], Tensor<TOut>> forward, Func<Tensor<TOut>, Value<TOut>, Tensor<float>> loss,
        Value<TIn>[][] inputs, Value<TOut>[] targets)
        where TIn : notnull where TOut : notnull
    {
        int aidx = inputs[0][0].AcceleratorIndex;
        using var totalLoss = new ScalarValue(0, aidx);
        for (int i = 0; i < inputs.Length; i++)
        {
            using var outputTensor = forward(inputs[i].Select(t => new Tensor<TIn>(t, t.Zeros())).ToArray());
            using var lossTensor = loss(outputTensor, targets[i]);
            totalLoss.UpdateWith(totalLoss + lossTensor.Value.AsScalar());
            compute.Flush(aidx);
        }
        return totalLoss.ToHost() / inputs.Length;
    }
    #endregion

    #region Serialization
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
    #endregion
}