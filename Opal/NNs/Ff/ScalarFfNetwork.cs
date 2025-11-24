using Opal.Autograd.Catalogs;
using Opal.Mathematics;
using Opal.Utilities;

namespace Opal.NNs.Ff;

public class ScalarFfNetwork : VectorFfNetwork
{
    public ScalarFfNetwork(
        int hiddenSize, int numHiddenLayers,
        Func<VectorTensor, VectorTensor> hiddenActivation,
        Func<VectorTensor, VectorTensor> outputActivation,
        Func<VectorTensor, VectorTensorStorage, ScalarTensor> lossFunction) :
        base(1, hiddenSize, 1, numHiddenLayers, hiddenActivation, outputActivation, lossFunction)
    {
    }
    
    public double Forward(double input) => Forward([input])[0];

    public VectorTensor Forward(ScalarTensor input) =>
        Forward(
            Operations.NewVector(
                Operations.VectorFromScalarStorage(input.Value), 
                null, _ => { }, Operations.NewDefaultVectorStorage(Vectors.Zeros(1))));

    public void Backwards(ScalarTensor input, ScalarTensor target, double learningRate = 0.01)
    {
        using var lr = Operations.NewDefaultScalarStorage(learningRate);
        using var output = Forward(input);
        using var loss = LossFunction(output, Operations.VectorFromScalarStorage(target.Value));
        loss.Backwards(Operations.NewScalar(1.0, 0.0));
        UpdateParameters(lr);
    }
        
}

