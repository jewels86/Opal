using Jewels.Lazulite;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : VectorFfNetwork // only use this if you can load layers- training would be difficult
{
    public ScalarFfNetwork(
        int hiddenSize, int numHiddenLayers,
        Func<Tensor<float[]>, Tensor<float[]>> hiddenActivation,
        Func<Tensor<float[]>, Tensor<float[]>> outputActivation,
        Func<Tensor<float[]>, Value<float[]>, Tensor<float>> lossFunction) :
        base(1, hiddenSize, 1, numHiddenLayers, hiddenActivation, outputActivation, lossFunction)
    {
    }
    
    public Value<float[]> Forward(Value<float> input) => Forward(new VectorValue(input.Data));
}

