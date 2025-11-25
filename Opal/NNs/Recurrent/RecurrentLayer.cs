using Jewels.Lazulite;

namespace Opal.NNs.Recurrent;

public class RecurrentLayer<TIn, TOut, TWeights> : ILayer<TIn, TOut>
    where TIn : notnull where TOut : notnull where TWeights : notnull
{
    public RecurrentLayer(
        Tensor<TWeights> inputWeights, Tensor<TWeights> recurrentWeights, 
        Tensor<TOut> biases, Tensor<TOut> state, 
        Func<Tensor<TOut>, Tensor<TOut>> activation, IRecurrentCatalog<TIn, TOut, TWeights> catalog)
    {
        InputWeights = inputWeights;
        RecurrentWeights = recurrentWeights;
        Biases = biases;
        State = state;
        Activation = activation;
        Catalog = catalog;
    }

    public Tensor<TWeights> InputWeights { get; set; } 
    public Tensor<TWeights> RecurrentWeights { get; set; }
    public Tensor<TOut> Biases { get; set; }
    public Tensor<TOut> State { get; set; }
    public Func<Tensor<TOut>, Tensor<TOut>> Activation { get; set; }
    public IRecurrentCatalog<TIn, TOut, TWeights> Catalog { get; set; }

    public Tensor<TOut> Forward(Tensor<TIn> input)
    {
        var inputPart = Catalog.Multiply(InputWeights, input);
        var hiddenPart = Catalog.Multiply(RecurrentWeights, State);
        var resultBuffer = inputPart.Value.Zeros();
        Compute.Call(inputPart.AcceleratorIndex, Operations.ElementwiseTripleAddKernels, inputPart.Value, hiddenPart.Value, Biases.Value, resultBuffer);
        var result = new Tensor<TOut>(
            inputPart.Value.Create(resultBuffer, inputPart.Value.Shape), 
            inputPart.Gradient.Zeros(), BackwardFunction, [inputPart, hiddenPart, Biases]);
        var output = Activation(result);
        State = output;
        return output;

        void BackwardFunction(ITensor t)
        {
            Compute.BinaryCall(Compute.ElementwiseAddKernels, t.Gradient.Data, inputPart.Gradient, inputPart.Gradient);
            Compute.BinaryCall(Compute.ElementwiseAddKernels, t.Gradient.Data, hiddenPart.Gradient, hiddenPart.Gradient);
            Compute.BinaryCall(Compute.ElementwiseAddKernels, t.Gradient.Data, Biases.Gradient, Biases.Gradient);
            inputPart.Dispose();
            hiddenPart.Dispose();
        }
    }

    public Value<TOut> Forward(Value<TIn> input) => Forward(new Tensor<TIn>(input, input.Zeros())).Value;

    public void UpdateParameters(float lr)
    {
        Compute.Call(InputWeights.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, InputWeights.Value, InputWeights.Value, InputWeights.Value, lr);
        Compute.Call(RecurrentWeights.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, RecurrentWeights.Value, RecurrentWeights.Value, RecurrentWeights.Value, lr);
        Compute.Call(Biases.AcceleratorIndex, Operations.ElementwiseFloatMulAndSubKernels, Biases.Value, Biases.Value, Biases.Value, lr);
        ZeroGradients();
    }

    public void ZeroGradients()
    {
        InputWeights.Gradient.UpdateWith(InputWeights.Gradient.Zeros());
        RecurrentWeights.Gradient.UpdateWith(RecurrentWeights.Gradient.Zeros());
        Biases.Gradient.UpdateWith(Biases.Gradient.Zeros());
    }

    public void Write(BinaryWriter writer)
    {
        Catalog.WriteWeights(writer, InputWeights.Value);
        Catalog.WriteWeights(writer, RecurrentWeights.Value);
        Catalog.WriteBias(writer, Biases.Value);
        Catalog.WriteState(writer, State.Value);
    }

    public void Read(BinaryReader reader)
    {
        var inputValue = Catalog.ReadWeights(reader);
        var recurrentValue = Catalog.ReadWeights(reader);
        InputWeights = new(inputValue, inputValue.Zeros());
        RecurrentWeights = new(recurrentValue, recurrentValue.Zeros());
        
        var biasValue = Catalog.ReadBias(reader);
        Biases = new(biasValue, biasValue.Zeros());
        
        var stateValue = Catalog.ReadState(reader);
        State = new(stateValue, stateValue.Zeros());
    }
}

public interface IRecurrentCatalog<TIn, TOut, TWeights>
    where TIn : notnull where TOut : notnull
    where TWeights : notnull
{
    public Tensor<TOut> Multiply(Tensor<TWeights> weights, Tensor<TIn> input);
    public Tensor<TOut> Multiply(Tensor<TWeights> weights, Tensor<TOut> state);
    
    public void WriteWeights(BinaryWriter writer, Value<TWeights> weight);
    public void WriteBias(BinaryWriter writer, Value<TOut> bias);
    public void WriteState(BinaryWriter writer, Value<TOut> state);
    
    public Value<TWeights> ReadWeights(BinaryReader reader);
    public Value<TOut> ReadBias(BinaryReader reader);
    public Value<TOut> ReadState(BinaryReader reader);
}